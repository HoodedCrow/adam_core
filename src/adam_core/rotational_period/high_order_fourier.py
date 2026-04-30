from ..orbits.query.horizons import RotationalPeriodInput
from .hg12star import hg12star_correction
from typing import Tuple, Optional
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from ..constants import DE44X_CONSTANTS
import scipy.optimize
import pickle
from .types import FourierFitResult, FourierFullResult

def _adjust_inputs_for_fourier(observation_table: pa.Table, range_inputs: RotationalPeriodInput,
                              obs_time: pa.DoubleArray, quiet: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Correct observations for the light time and distance.
    Returns corrected observed times and magnitudes in the original order."""
    # Sanity that the arrays are sorted, so we can just match indexes
    deltas = pc.subtract(obs_time, range_inputs.obs_time.mjd())
    assert pc.max(pc.abs(deltas)).as_py() < 1e-9

    observed_mags = np.array([float(v) for v in observation_table["mag"]])
    helio_r = range_inputs.sun_range.to_numpy()
    topo_delta = range_inputs.obs_range.to_numpy()
    alpha = range_inputs.angle.to_numpy()
    if not quiet:
        print(f"Alpha range {np.min(alpha)} - {np.max(alpha)}")

    # Page 6, section 3.1
    # Subtract 5*log10(r*delta) from observational magnitudes
    # print(f"Sizes V={observed_mags.shape}, r={helio_r.shape}, delta={topo_delta.shape}, alpha={alpha.shape}")
    observed_mags = observed_mags - 5*np.log10(helio_r * topo_delta)

    # Correct timing for speed of light: t = t_obs - delta/c
    observed_time = obs_time - topo_delta / DE44X_CONSTANTS["C"]
    return observed_time, observed_mags

def _get_freqs(fmin: float, fmax: float, arc_len: float) -> np.ndarray:
    """Returns the list of frequencies to try in Fourier analysis."""
    N = int(30 * arc_len * (fmax - fmin))
    return np.linspace(fmin, fmax, num=N)

def _fit_fourier(freq: float, k: int, obs_v, obs_t, alpha_deg, bands, weights, kind: Optional[int]) -> Tuple[np.ndarray, float, int]:
    # kind == None means use c1,c2. kind < 0 means fit G12*, kind >= 0 selects from G_VALUES
    # Returned values are [c1, c2, A1, B1, ..., Ak, Bk, H_g, H_i, H_r, H_u], length 2+2*k+4
    # In case G12* fit, c1=1e6, c2=G12*. In case known G1,G2, c1=index of G_VALUES and c2=1e6

    # return values, sigma2, np.sum(included)

    use_g12star = kind is not None and kind < 0

    # Figure out num parameters and size of matrices
    num_obs = len(obs_v)
    if use_g12star:
        num_params = 1 + 2*k + 4 # include g12*
        A = np.zeros((num_obs, num_params-1))
    else:
        if kind is not None:
            num_params = 2*k + 4
            obs_v = obs_v - hg12star_correction(alpha_deg, kind=kind)
        else:
            num_params = 2 + 2*k + 4 # include c1, c2
        A = np.zeros((num_obs, num_params))
    params = np.zeros(num_params)

    # Populate A matrix based on the choice of alpha dependency function
    alpha_rad = np.deg2rad(alpha_deg)
    if kind is None:
        # Model phi(alpha) = c1*alpha + c2*alpha^2
        # X=[c1, c2, A1, B1, ..., Ak, Bk, Hg, Hi, Hr, Hu ]
        A[:, 0] = alpha_rad # *c1
        A[:, 1] = alpha_rad*alpha_rad # *c2
        idx = 2
    else:
        # Either known kind or fitting g12*. In both cases nothing goes into
        # the A matrix except Fourier cooeficients and H
        idx = 0

    # Fourier coefficients for
    # g (t ) = sum(j=1:k) Aj*cos (2 fjt ) + Bj*sin (2 fjt ) for k 2:6
    base = 2*np.pi*freq*obs_t
    for j in range(1, k+1):
        arg = base * j
        A[:, idx] = np.cos(arg)
        A[:, idx+1] = np.sin(arg)
        idx+=2

    # Pick color. TODO: use matrix operations here
    color_offset = {'g': 0, 'i': 1, 'r': 2, 'u': 3}
    for i in range(num_obs):
        color = bands[i].as_py()
        try:
            A[i, idx+color_offset[color]] = 1
        except KeyError:
            raise BaseException(f"Bad color [{color}] index {i}")

    # Build weighted matrices. obs_v is already adjusted if we use known G1,G2
    W = np.diag(np.sqrt(weights))
    Aw = np.dot(W, A)
    Bw = np.dot(obs_v, W)

    # Function for computing residuals if using non-linear least squares
    # This is used only if we have g12star part, but make it generic unless
    # that if is somehow computationally expensive (compared to the matrix ops -
    # doubt it)
    included = np.array([True] * num_obs)
    def func(par):
        if use_g12star:
            corr = hg12star_correction(alpha_deg, g12star=par[0])
            return Aw[included, :] @ par[1:] + np.dot(corr, W)[included] - Bw[included]
        return Aw[included, :] @ par - Bw[included]

    # May need to throw away outliers
    converged = False
    while not converged:
        # Use regular least squares unless we have to fit g12star
        if use_g12star:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
            lb = [-np.inf] * len(params)
            ub = [np.inf] * len(params)
            lb[0], ub[0] = 0.0, 1.0 # bounds for g12*
            result = scipy.optimize.least_squares(func, params, bounds=(lb, ub), verbose=0)
            if not result.success:
                print(f"Non-linear least squares failed:\n{result}")
                return np.zeros(2 + 2*k + 4), -1, -1
            values = result.x
            print(f"Mask {result.active_mask}")
        else:
            values, _residuals, _rank, _singulars = np.linalg.lstsq(Aw[included, :], Bw[included])

        # Compute p_j*(Oj-Cj)^2
        # This includes already rejected, it's okay
        if use_g12star:
            corr = hg12star_correction(alpha_deg, g12star=values[0])
            res = (A @ values[1:] + corr - obs_v)**2 * weights
        else:
            res = (A @ values - obs_v)**2 * weights 
        n_incl = np.sum(included)
        # Discard rejected when computing sigma^2
        sigma2 = np.sum(res[included]) / (n_incl - num_params) 
        # No masking needed here, because included accumulates
        outliers = res > 9*sigma2
        # Converged if there are no NEW outliers
        converged = np.sum(outliers[included]) == 0
        included = included & ~outliers

    # Pad values for G12* cases
    if use_g12star:
        values = np.concatenate([[1e6], values])
    elif kind is not None:
        values = np.concatenate([[kind, 1e6], values])
    return values, num_obs * sigma2 / np.sum(weights), np.sum(included)


def run_fourier(observation_table: pa.Table, range_inputs: RotationalPeriodInput, obs_time: pa.DoubleArray,
                kind: Optional[int], freqs) -> Tuple[FourierFitResult, float]:
    observed_time, observed_mags = _adjust_inputs_for_fourier(observation_table, range_inputs, obs_time)
    arc_length = np.max(obs_time) - np.min(obs_time)
    print(f"Freqs count {freqs.shape}, arc length={arc_length}")

    alpha = np.deg2rad(range_inputs.angle.to_numpy())
    weights = 1 / (np.array([float(w) for w in observation_table["rmsmag"]]) ** 2)

    # Page 7: for each k, find freq that gives minimum sigma2 for that k
    rows_k, rows_sigma2, rows_freq, rows_n_included, rows_values = [], [], [], [], []
    for k in range(2, 7):
        best_sigma2 = np.inf
        best_fit = None
        best_freq = np.inf
        best_size = np.inf
        idx = 0
        for f in freqs:
            if kind is not None and kind < 0:
                print(f"k={k} idx={idx} out of {len(freqs)}")
                idx += 1
            X, sigma2, n_incl = _fit_fourier(f, k, observed_mags, observed_time, alpha, observation_table["band"], weights, kind)
            if sigma2 < best_sigma2:
                best_sigma2 = sigma2
                best_fit = X
                best_freq = f
                best_size = n_incl
        print(f"For k={k} got best freq {best_freq} with sigma={best_sigma2} H={best_fit[-4:]} N={best_size} -> period = {24.0 / best_freq} h")
        rows_k.append(k)
        rows_sigma2.append(best_sigma2)
        rows_freq.append(best_freq)
        rows_n_included.append(int(best_size))
        rows_values.append(best_fit.tolist())

    return FourierFitResult.from_kwargs(
        k=rows_k,
        sigma2=rows_sigma2,
        freq=rows_freq,
        n_included=rows_n_included,
        values=rows_values,
    ), arc_length

def best_row_by_f(fourier_result: FourierFitResult, alpha: float=0.05) -> FourierFitResult:
    """Select the best Fourier order k using an F-test.
    Returns the single winning row as a FourierFitResult."""
    best_row = 0
    best_row_changed = True
    while best_row_changed:
        best_row_changed = False
        k_i      = fourier_result.k[best_row].as_py()
        sigma2_i = fourier_result.sigma2[best_row].as_py()
        size_i   = fourier_result.n_included[best_row].as_py()
        for j in range(best_row + 1, len(fourier_result)):
            k_j      = fourier_result.k[j].as_py()
            sigma2_j = fourier_result.sigma2[j].as_py()
            size_j   = fourier_result.n_included[j].as_py()
            F = sigma2_i / sigma2_j
            p = 1 - scipy.stats.f.cdf(F, size_i - 1, size_j - 1)
            print(f"K={k_i},{k_j} sigmas2={sigma2_i},{sigma2_j} F={F}, p={p}")
            if p < alpha and sigma2_j < sigma2_i:
                print(f"Switch best row to {j}")
                best_row = j
                best_row_changed = True
                break
    best = fourier_result.take([best_row])
    print(f"Best row={best_row} for k={best.k[0].as_py()}")
    return best


def amplitude(fit: FourierFitResult, alpha_avg: float) -> Tuple[float, float, float, float, float, int]:
    """Compute amplitude, colors, and elongation from the best-fit Fourier row.

    Parameters
    ----------
    fit : single-row FourierFitResult (the winner from best_row_by_f)
    alpha_avg : mean phase angle [degrees]

    Returns
    -------
    ampl, color_gr, color_gi, color_ri, elongation, count_local_maxima
    """
    k      = fit.k[0].as_py()
    freq   = fit.freq[0].as_py()
    values = fit.values[0].as_py()  # [c1, c2, A1, B1, ..., Ak, Bk, H_g, H_i, H_r]

    period_days = 1 / freq
    sample = np.linspace(0, 2*period_days, num=1000)
    base = 2*np.pi*freq*sample
    G = np.zeros(len(sample))
    assert len(values) == 2 + k*2 + 4
    for j in range(k):
        arg = base * (j + 1)
        G += values[2+2*j]   * np.cos(arg)
        G += values[2+2*j+1] * np.sin(arg)
    ampl = np.max(G) - np.min(G)

    max_count = sum(1 for i in range(1, len(G)-1) if G[i-1] < G[i] and G[i] >= G[i+1])
    print(f"Count of local maxima {max_count}, G size {len(G)}")
    print(f"Amplitude {ampl}")

    H_g = values[2+2*k]
    H_i = values[2+2*k+1]
    H_r = values[2+2*k+2]
    assert 2+2*k+3 == len(values)-1
    print(f"Colors g-r={H_g-H_r} g-i={H_g-H_i} r-i={H_r-H_i}")

    A0 = ampl / (1 + 0.02*alpha_avg)
    elongation = 10 ** (0.4*A0)
    print(f"Elongation={elongation}")
    return ampl, H_g-H_r, H_g-H_i, H_r-H_i, elongation, max_count

def run_complete_fourier(object_id: str, selection, obstime, inputs, kind: Optional[int], min_freq: float = 0.1667, max_freq: float = 500, preselect_freq: bool = True) -> FourierFullResult:
    """Run the complete Fourier rotational-period pipeline for one object.

    Returns a single-row FourierFullResult with the best-fit summary
    and the winning FourierFitResult row embedded as intermediate_result.
    """
    print(f"All colors {selection['band'].unique().to_pylist()}")

    arc_length = np.max(obstime) - np.min(obstime)
    freqs = _get_freqs(min_freq, max_freq, arc_length)
    if kind is not None and kind < 0 and preselect_freq:
        g12star_helper_kind = 0
        print(f"Computing frequency first to make G12* fitting easier, use type {g12star_helper_kind}") # = '{G_VALUES[g12star_helper_kind][0]}'")
        fit, arc_length = run_fourier(selection, inputs, obstime, g12star_helper_kind, freqs)
        best_fit = best_row_by_f(fit)
        freq = best_fit.freq[0].as_py()
        print(f"Selected frequency {freq} for period {24.0 / freq} h")
        fit, arc_length = run_fourier(selection, inputs, obstime, kind, np.array([freq]))
    else:
        fit, arc_length = run_fourier(selection, inputs, obstime, kind, freqs)

    best_fit = best_row_by_f(fit)
    ampl, color_gr, color_gi, color_ri, elongation, max_count = amplitude(
        best_fit, np.average(inputs.angle)
    )
    print(f"Angle in degrees: average={np.average(inputs.angle)}, mean={np.mean(inputs.angle)}, median={np.median(inputs.angle)}")
    period_h = 24.0 / best_fit.freq[0].as_py()
    if max_count < 3:
        print(f"Double the period, count of maximums is {max_count}")
        period_h *= 2
    return FourierFullResult.from_kwargs(
        object_id=[object_id],
        num_obs = [len(obstime)],
        arc_days = [arc_length],
        intermediate_result=[pickle.dumps(fit)],
        selected_k=[best_fit.k[0].as_py()],
        count_local_maxima=[max_count],
        period_h=[period_h],
        amplitude=[ampl],
        color_gr=[color_gr],
        color_gi=[color_gi],
        color_ri=[color_ri],
        elongation=[elongation],
    )