from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from astropy.time import Time
from mpcq import MPCObservations
from mpcq.orbits import MPCOrbits

from adam_core.orbits.query.horizons import (
    RotationalPeriodInput,
    cache_or_query_rotational_period_inputs,
)
from adam_core.rotational_period.high_order_fourier import run_complete_fourier
from adam_core.time import Timestamp

all_data = pq.read_table("data/paper_data.parquet")
all_data = all_data.sort_by("obstime")


@pytest.fixture
def mm81_data():
    object_id = "2025 MM81"
    observ = all_data.filter(pc.equal(all_data["provid"], object_id))
    obstime = Timestamp.from_astropy(Time(observ["obstime"].to_pylist())).mjd()
    inputs = cache_or_query_rotational_period_inputs(
        object_id, "X05", obstime, "data/rot_period_inputs.parquet"
    )
    return object_id, observ, obstime, inputs


@pytest.mark.benchmark(group="rotper_speed")
# @pytest.mark.parametrize(
#     "fitter",
#     [find_orb_fitter, orbfit_orb_fitter, layup_orb_fitter],
#     ids=lambda val: f"{type(val).__name__}",
# )
# @pytest.mark.parametrize("object_id", objects)
def test_initial_orbit_fit_benchmark(benchmark, mm81_data):
    object_id, obs_table, obstime, inputs = mm81_data
    result = benchmark(
        run_complete_fourier, object_id, obs_table, obstime, inputs, kind=None
    )
    assert (
        result.object_id[0].as_py() == object_id
    ), f"Wrong object_id '{result.object_id[0].as_py()}'"
    assert (
        result.num_obs[0].as_py() == 390
    ), f"Num obs is {result.num_obs[0]}, expected 390"

    def check_num(actual, expected, tolerance, message):
        assert (
            abs(actual - expected) < tolerance
        ), f"{message} is {actual}, expected {expected}"

    check_num(result.arc_days[0].as_py(), 9, 1.0, "Arc days")
    check_num(result.period_h[0].as_py(), 1.1, 0.1, "Period h")
    check_num(result.amplitude[0].as_py(), 1.2, 0.2, "Amplitude")
    check_num(result.elongation[0].as_py(), 2.1, 0.1, "Elongation")
    check_num(result.color_gr[0].as_py(), 0.53, 0.1, "g-r")
    check_num(result.color_gi[0].as_py(), 0.66, 0.1, "g-i")
    check_num(result.color_ri[0].as_py(), 0.12, 0.1, "r-i")
