import pyarrow as pa
import quivr as qv

from ..time import Timestamp


class RotationalPeriodPhotometry(qv.Table):
    """Input observations for rotational period calculation."""

    object_id = qv.LargeStringColumn()
    stn = qv.LargeStringColumn()
    obs_time = Timestamp.as_column()
    band = qv.LargeStringColumn()
    mag = qv.Float64Column()
    rmsmag = qv.Float64Column()


class FourierFitResult(qv.Table):
    """One row per Fourier order k (2–6) tried by run_fourier.
    Stores the best-fit frequency and residual found for that k."""

    k = qv.Int64Column(nullable=True)  # Fourier order
    sigma2 = qv.Float64Column(
        nullable=True
    )  # best weighted residual (minimized over freq grid)
    freq = qv.Float64Column(
        nullable=True
    )  # best frequency [cycles/day]; period_h = 24/freq
    n_included = qv.Int64Column(
        nullable=True
    )  # observations retained after outlier rejection
    # Fit coefficients [c1, c2, A1, B1, ..., Ak, Bk, H_g, H_i, H_r, H_u], length = 2+2k+4
    # In case G12* fit, c1=1e6, c2=G12*. In case known G1,G2, c1=index of G_VALUES and c2=1e6
    values = qv.LargeListColumn(pa.float64(), nullable=True)


class FourierFullResult(qv.Table):
    """Complete output of Fourier fit"""

    object_id = qv.LargeStringColumn()
    # Total number of observations available, with outliers
    num_obs = qv.Int64Column(nullable=True)
    arc_days = qv.Float64Column(nullable=True)
    # Multiple rows of FourierFitResult
    intermediate_result = qv.BinaryColumn(nullable=True)
    selected_k = qv.Int64Column(nullable=True)
    count_local_maxima = qv.Int64Column(nullable=True)
    period_h = qv.Float64Column(nullable=True)
    amplitude = qv.Float64Column(nullable=True)
    color_gr = qv.Float64Column(nullable=True)
    color_gi = qv.Float64Column(nullable=True)
    color_ri = qv.Float64Column(nullable=True)
    elongation = qv.Float64Column(nullable=True)
    runtime = qv.Int64Column(nullable=True)
    method = qv.LargeStringColumn(nullable=True)
