import logging
import os

import pyarrow as pa
import pyarrow.compute as pc
import quivr as qv
from astropy.time import Time
from google.cloud import bigquery

from adam_core.time import Timestamp

from .types import RotationalPeriodPhotometry

logger = logging.getLogger(__name__)


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
    if cache_file is not None:
        try:
            output_data = RotationalPeriodPhotometry.from_parquet(cache_file)
            logger.debug(f"Read total {len(output_data)} records")
        except:
            print(f"Failed to read {cache_file}")
            output_data = RotationalPeriodPhotometry.empty()
    else:
        output_data = RotationalPeriodPhotometry.empty()
    mask = pc.and_(
        pc.equal(output_data.object_id, object_id), pc.equal(output_data.stn, stn)
    )
    result = output_data.apply_mask(mask)
    logger.error(f"Got {len(result)} records out of {len(output_data)}")
    if force_reload or len(result) == 0:
        logger.error(f"Reloading for object {object_id}")
        result = _query_rotational_period_photometry_inputs(object_id, stn, dataset_id)
        result = result.sort_by(["obs_time"])
        if cache_file is not None:
            output_data = qv.concatenate(
                [output_data.apply_mask(pc.invert(mask)), result]
            )
            output_data.to_parquet(cache_file)
    return result
