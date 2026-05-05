import time
from typing import Optional, Tuple

import numpy as np
import pyarrow.compute as pc
import quivr as qv
from gatspy.periodic import LombScargleMultiband

from .types import FourierFullResult, RotationalPeriodPhotometry

GATSPY_METHOD = "LSM gatspy"


def lsm_amplitude(g_coeff: np.ndarray, period_days: float) -> Tuple[float, int]:
    sample = np.linspace(0, 2 * period_days, num=1000)
    base = 2 * np.pi / period_days * sample
    G = np.zeros(len(sample))
    for j in range(2):
        arg = base * (j + 1)
        G += g_coeff[2 * j] * np.cos(arg)
        G += g_coeff[2 * j + 1] * np.sin(arg)
    ampl = np.max(G) - np.min(G)
    max_count = sum((np.roll(G, -1) < G) & (np.roll(G, 1) <= G))
    return ampl, max_count


def run_lsm(
    photometry: RotationalPeriodPhotometry,
    # mag_v: float,
    p_min: float = 0.00065,
    p_max: float = 3.0,
):
    # This is what paper says. We are not doing any of this for gatspy.
    # subtract MPC-predicted v-band magnitude
    mag = photometry.mag.to_numpy()  # - mag_v
    # normalize by median
    # print(f"Median for normalization {np.median(mag)}")
    # mag = mag / np.median(mag)
    # remove outliers more than 3 standard deviations away

    obstime = photometry.obs_time.mjd()
    arc_days = np.max(obstime) - np.min(obstime)
    # dense, uniform grid search for the best periodic signal with a
    # range between 0.00065 days (0.936 minutes) and 3 days (72 hr)
    n_periods = int((1 / p_min - 1 / p_max) * arc_days * 100)
    print(f"Num periods {n_periods}")

    model = LombScargleMultiband(Nterms_base=2, Nterms_band=0)
    model.ﬁt(obstime, mag, photometry.rmsmag, photometry.band)
    # Compute power at the following periods, in days
    periods = np.linspace(p_min, p_max, n_periods)
    power = model.periodogram(periods)

    sanity_countdown = n_periods / 10
    while sanity_countdown > 0:
        sanity_countdown -= 1
        peak_idx = np.argmax(power)
        period = periods[peak_idx]
        print(f"Best period={period} days = {period * 24} h")
        # Base: H0, A1, B1, A2, B2; for each color Hc
        params = model._best_params(2 * np.pi / period)
        amplitude, max_count = lsm_amplitude(params[1:5], period)
        print(
            f"Ampl {amplitude}, max_count {max_count}. All {len(params)} params {params}"
        )
        if max_count > 3:
            break
        # Wipe this peak and find next one
        power[peak_idx] = 0

    # base_offset = params[0]
    # color_offsets = params[-3:]
    # r_color_off = params[-3]
    # g_color_off = params[-2]
    # i_color_off = params[-1]
    # print("Deltas between color offsets ", g_color_off - r_color_off,
    #       g_color_off - i_color_off, r_color_off - i_color_off)

    # print(r_color_off, g_color_off, i_color_off)
    # print("Model colors", model.unique_filts_)
    # print("Y mean by filter", model.ymean_by_filt_)
    # g_color = model.ymean_by_filt_[0]
    # r_color = model.ymean_by_filt_[2]
    # i_color = model.ymean_by_filt_[1]
    # print("Deltas between means", g_color - r_color, g_color - i_color, r_color - i_color)
    # print(r_color, g_color, i_color)

    # print(g_color - r_color + (g_color_off - r_color_off))

    r_band = model.predict(obstime, ["r"], period)
    i_band = model.predict(obstime, ["i"], period)
    g_band = model.predict(obstime, ["g"], period)
    # g_color = np.average(g_band)
    # r_color = np.average(r_band)
    # i_color = np.average(i_band)
    # print("Deltas between predictions", g_color - r_color, g_color - i_color, r_color - i_color)

    color_gr = np.average(g_band - r_band)
    color_gi = np.average(g_band - i_band)
    color_ri = np.average(r_band - i_band)

    return FourierFullResult.from_kwargs(
        object_id=photometry.object_id.unique(),
        num_obs=[len(photometry)],
        arc_days=[arc_days],
        period_h=[24 * period],
        amplitude=[amplitude],
        color_gr=[color_gr],
        color_gi=[color_gi],
        color_ri=[color_ri],
        method=[GATSPY_METHOD],
    )


def run_gatspy_lsm_cached(
    object_id: str,
    photometry: RotationalPeriodPhotometry,
    # mag_v: float,
    cache_file: Optional[str],
    force_reload: bool = False,
):
    try:
        # Let it throw on None
        output_data = FourierFullResult.from_parquet(cache_file or "")
        print(f"Read total {len(output_data)} records from {cache_file}")
    except:
        print(f"Failed to read {cache_file}")
        output_data = FourierFullResult.empty()
    mask = pc.and_(
        pc.equal(output_data.object_id, object_id),
        pc.equal(output_data.method, GATSPY_METHOD),
    )
    result = output_data.apply_mask(mask)
    print(f"Got {len(result)} records out of {len(output_data)}")
    if force_reload or len(result) == 0:
        print(f"Recomputing Gatspy LSM for object {object_id}")
        start = time.perf_counter_ns()
        result = run_lsm(photometry)  # , mag_v)
        runtime = time.perf_counter_ns() - start
        result = result.set_column("runtime", [runtime])
        if cache_file is not None:
            output_data = qv.concatenate(
                [output_data.apply_mask(pc.invert(mask)), result]
            )
            output_data.to_parquet(cache_file)
    else:
        print(f"Return cached result for {object_id} method {GATSPY_METHOD}")
    return result
