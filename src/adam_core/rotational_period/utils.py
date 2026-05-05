import logging
import os
import time
from typing import Callable, Optional

import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from astropy.time import Time
from google.cloud import bigquery

from adam_core.time import Timestamp

from ..orbits.query.horizons import (
    RotationalPeriodInput,
    query_rotational_period_inputs_from_horizons,
)
from .types import FourierFullResult, RotationalPeriodPhotometry

logger = logging.getLogger(__name__)


def cache_table(
    table_type,
    cache_file: Optional[str],
    mask_fn: Callable,
    compute_fn: Callable,
    transform_fn: Optional[Callable] = None,
    force_reload: bool = False,
    verbose: bool = True,
):
    """Generic load-filter-compute-writeback pattern for quivr table caches.

    Parameters
    ----------
    table_type   : quivr Table subclass with from_parquet() and empty() class methods
    cache_file   : path to the parquet cache (None skips caching)
    mask_fn      : (table) -> bool array selecting the cached rows for this key
    compute_fn   : () -> table   — called on a cache miss
    transform_fn : optional (result) -> result applied to a fresh result before caching
    force_reload : ignore the cache and always recompute
    verbose      : print progress messages
    """
    try:
        data = table_type.from_parquet(cache_file or "")
        if verbose:
            print(f"Read total {len(data)} records from {cache_file}")
    except Exception:
        if verbose:
            print(f"Failed to read {cache_file}")
        data = table_type.empty()

    mask = mask_fn(data)
    result = data.apply_mask(mask)
    if verbose:
        print(f"Got {len(result)} records out of {len(data)}")

    if force_reload or len(result) == 0:
        result = compute_fn()
        if transform_fn is not None:
            result = transform_fn(result)
        if cache_file is not None:
            data = qv.concatenate([data.apply_mask(pc.invert(mask)), result])
            data.to_parquet(cache_file)
    # callers print their own cache-hit message if needed

    return result


def run_cached(
    object_id: str,
    method: str,
    cache_file: Optional[str],
    compute_fn: Callable[[], FourierFullResult],
    force_reload: bool = False,
) -> FourierFullResult:
    """Load a FourierFullResult from cache or compute it, stamping runtime and method."""

    def _timed_compute():
        start = time.perf_counter_ns()
        result = compute_fn()
        runtime = time.perf_counter_ns() - start
        result = result.set_column("runtime", [runtime])
        result = result.set_column("method", pa.array([method], type=pa.large_string()))
        print(f"Computed {method} for {object_id} in {runtime / 1e9:.1f}s")
        return result

    return cache_table(
        FourierFullResult,
        cache_file,
        lambda d: pc.and_(pc.equal(d.object_id, object_id), pc.equal(d.method, method)),
        _timed_compute,
        force_reload=force_reload,
    )


def _query_rotational_period_photometry_inputs(
    object_id: str, stn: str, dataset_id: str
):
    query = f"""SELECT obs.obstime, obs.mag, obs.rmsmag, obs.band 
                FROM {dataset_id}.public_obs_sbn AS obs
                WHERE obs.stn = '{stn}' AND obs.provid = '{object_id}' AND obs.rmsmag IS NOT NULL;"""
    client = bigquery.Client(project=os.environ["MPCQ_PROJECT_ID"])
    query_job = client.query(query)
    results = query_job.result()
    table = results.to_arrow(progress_bar_type="tqdm", create_bqstorage_client=True)
    band = table["band"]
    mag = table["mag"].cast(pa.float64())
    rmsmag = table["rmsmag"].cast(pa.float64())
    obstime = Timestamp.from_astropy(Time(table["obstime"].to_pylist()))  # .mjd()
    return RotationalPeriodPhotometry.from_kwargs(
        object_id=[object_id] * len(band),
        stn=[stn] * len(band),
        obs_time=obstime,
        band=band,
        mag=mag,
        rmsmag=rmsmag,
    )


def get_rotational_period_photometry_inputs(
    object_id: str,
    stn: str,
    cache_file: str,
    dataset_id: str,
    force_reload: bool = False,
):
    return cache_table(
        RotationalPeriodPhotometry,
        cache_file,
        lambda d: pc.and_(pc.equal(d.object_id, object_id), pc.equal(d.stn, stn)),
        lambda: _query_rotational_period_photometry_inputs(object_id, stn, dataset_id),
        transform_fn=lambda r: r.sort_by(["obs_time"]),
        force_reload=force_reload,
    )


def cache_or_query_rotational_period_inputs(
    object_id: str,
    stn: str,
    obs_times: pa.DoubleArray,
    cache_file: str,
    force_reload: bool = False,
    quiet: bool = False,
) -> RotationalPeriodInput:
    return cache_table(
        RotationalPeriodInput,
        cache_file,
        lambda d: pc.equal(d.object_id, object_id),
        lambda: query_rotational_period_inputs_from_horizons(object_id, stn, obs_times),
        force_reload=force_reload,
        verbose=not quiet,
    )
