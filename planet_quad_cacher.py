"""Tracer code to set up training pipeline."""
import os
import logging
import multiprocessing
import pathlib
import queue
import sqlite3
import subprocess
import threading

from osgeo import gdal
from osgeo import osr
import ecoshard
import pygeoprocessing
import requests
import retrying
import shapely.geometry
import shapely.ops
import shapely.prepared
import shapely.wkb
import taskgraph

WORKSPACE_DIR = 'planet_quad_cache_workspace'
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
QUAD_DIR = os.path.join(CHURN_DIR, 'quads')
DATABASE_PATH = os.path.join(WORKSPACE_DIR, 'quad_uri.db')
N_WORKERS = multiprocessing.cpu_count()

COUNTRY_BORDER_VECTOR_URI = (
    'gs://natgeo-dams-data/ecoshards/'
    'countries_iso3_md5_6fb2431e911401992e6e56ddf0a9bcda.gpkg')

# This is the Planet Mosaic ID for global_quarterly_2019q2_mosaic
MOSAIC_ID = '4ce5863a-fb3f-4cad-a899-b8c053af1858'
PLANET_API_KEY_FILE = 'planet_api_key.txt'
REQUEST_TIMEOUT = 1.5

_WGS84_SRS = osr.SpatialReference()
_WGS84_SRS.ImportFromEPSG(4326)
WGS84_WKT = _WGS84_SRS.ExportToWkt()

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)

@retrying.retry(wait_exponential_multiplier=100, wait_exponential_max=1000)
def _database_manager(database_path, database_command_queue):
    """Manage a database by accepting commands through the queue."""
    with sqlite3.connect(database_path) as connection:
        while True:
            try:
                payload = database_command_queue.get()
                if payload == 'STOP':
                    break
                sqlite_command, argument_list, mode, execute = payload
                if execute == 'execute':
                    cursor = connection.execute(sqlite_command, argument_list)
                elif execute == 'many':
                    cursor = connection.executemany(
                        sqlite_command, argument_list)
                elif execute == 'script':
                    cursor = connection.executescript(sqlite_command)
                else:
                    raise ValueError('Unknown execute mode: %s' % execute)
                cursor.close()
                connection.commit()
            except Exception:
                LOGGER.exception("error in _database_manager")
                raise

@retrying.retry(wait_exponential_multiplier=100, wait_exponential_max=1000)
def _execute_sqlite(
        sqlite_command, database_path, database_command_queue=None,
        argument_list=None, mode='read_only', execute='execute', fetch=None):
    """Execute SQLite command and attempt retries on a failure.

    Args:
        sqlite_command (str): a well formatted SQLite command.
        database_path (str): path to the SQLite database to operate on.
        database_command_queue (Queue): if not None, send this command along
            onto a managed database connection.
        argument_list (list): `execute == 'execute` then this list is passed to
            the internal sqlite3 `execute` call.
        execute (str): must be either 'execute', 'many', or 'script'.
        fetch (str): if not `None` can be either 'all' or 'one'.
            If not None the result of a fetch will be returned by this
            function.

    Returns:
        result of fetch if `fetch` is not None.

    """
    cursor = None
    connection = None
    if argument_list is None:
        argument_list = []
    try:
        if fetch:
            ro_uri = r'%s?mode=ro' % pathlib.Path(
                os.path.abspath(database_path)).as_uri()
            LOGGER.debug(
                '%s exists: %s', ro_uri, os.path.exists(os.path.abspath(
                    database_path)))
            connection = sqlite3.connect(ro_uri, uri=True)
        elif database_command_queue:
            database_command_queue.put(
                (sqlite_command, argument_list, mode, execute))
            return
        else:
            connection = sqlite3.connect(database_path)

        if execute == 'execute':
            cursor = connection.execute(sqlite_command, argument_list)
        elif execute == 'many':
            cursor = connection.executemany(sqlite_command, argument_list)
        elif execute == 'script':
            cursor = connection.executescript(sqlite_command)
        else:
            raise ValueError('Unknown execute mode: %s' % execute)

        result = None
        payload = None
        if fetch == 'all':
            payload = (cursor.fetchall())
        elif fetch == 'one':
            payload = (cursor.fetchone())
        elif fetch is not None:
            raise ValueError('Unknown fetch mode: %s' % fetch)
        if payload is not None:
            result = list(payload)
        cursor.close()
        connection.commit()
        connection.close()
        return result
    except Exception:
        LOGGER.exception('Exception on _execute_sqlite: %s', sqlite_command)
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.commit()
            connection.close()
        raise


def create_status_database(database_path):
    """Create a runtime status database if it doesn't exist.

    Args:
        database_path (str): path to database to create.

    Returns:
        None.
    """
    LOGGER.debug('launching create_status_database')
    create_database_sql = (
        """
        CREATE TABLE IF NOT EXISTS quad_cache_table (
            quad_id TEXT NOT NULL PRIMARY KEY,
            long_min FLOAT NOT NULL,
            lat_min FLOAT NOT NULL,
            long_max FLOAT NOT NULL,
            lat_max FLOAT NOT NULL,
            file_size INTEGER NOT NULL,
            gs_uri TEXT NOT NULL
            );

        CREATE INDEX IF NOT EXISTS long_min_index
            ON quad_cache_table (long_min);
        CREATE INDEX IF NOT EXISTS lat_min_index
            ON quad_cache_table (lat_min);
        CREATE INDEX IF NOT EXISTS long_max_index
            ON quad_cache_table (long_max);
        CREATE INDEX IF NOT EXISTS lat_max_index
            ON quad_cache_table (lat_max);

        CREATE TABLE IF NOT EXISTS processed_grid_table (
            grid_id TEXT NOT NULL PRIMARY KEY,
            long_min FLOAT NOT NULL,
            lat_min FLOAT NOT NULL,
            long_max FLOAT NOT NULL,
            lat_max FLOAT NOT NULL,
            status TEXT NOT NULL
            );
        """)
    if os.path.exists(database_path):
        os.remove(database_path)
    connection = sqlite3.connect(database_path)
    connection.executescript(create_database_sql)
    connection.commit()
    connection.close()


def fetch_quad_worker(
        work_queue, planet_api_key, quad_database_path, cache_dir,
        database_command_queue, global_report_queue):
    """Pull tuples from `work_queue` and process.

    Args:
        work_queue (queue): Expect work or STOP from here. If work payload is
            (mosaic_id, grid_id, long_min, lat_min, long_max, lat_max,
             quad_id_list)
        planet_api_key (str): API key for Planet
        quad_database_path (str): path to main sqlite status database.
        cache_dir (str): path to directory to allow local copy of files
        database_command_queue (queue): used to pass to _execute_sqlite so
            it can do so in an asynchronous manner
        global_report_queue (queue): used to report when quads are uploaded.

    Returns:
        None.
    """
    session = requests.Session()
    session.auth = (planet_api_key, '')

    while True:
        payload = work_queue.get()
        LOGGER.debug(f'fetch_quad_worker payload: {payload}')
        if payload == 'STOP':
            work_queue.put(payload)
            break
        LOGGER.debug(f'this is the payload: {payload}')
        (mosaic_id, grid_id, long_min, lat_min, long_max, lat_max,
         quad_id_list) = payload
        global_report_queue.put(
            (grid_id, long_min, lat_min, long_max, lat_max, len(quad_id_list)))

        thread_list = []
        to_copy_queue = queue.Queue()
        for quad_id in quad_id_list:
            fetch_worker_thread = threading.Thread(
                target=fetch_quad,
                args=(
                    session, quad_database_path, mosaic_id, quad_id, cache_dir,
                    to_copy_queue, global_report_queue, grid_id))
            fetch_worker_thread.start()
            thread_list.append(fetch_worker_thread)

        copy_quad_to_bucket_worker_thread = threading.Thread(
            target=_copy_quad_to_bucket_worker,
            args=(grid_id, quad_database_path, database_command_queue,
                  to_copy_queue, global_report_queue))
        copy_quad_to_bucket_worker_thread.start()

        for thread in thread_list:
            LOGGER.debug(f"joining thread {thread}")
            thread.join()


@retrying.retry()
def _global_grid_recorder(
        global_processing_map, quad_database_path, database_command_queue,
        global_report_queue):
    """Worker to keep track of what's done at a global grid scale.

    Args:
        global_processing_map (dict): a global dict that can be used to index
            by `grid_id` into a list containing an element of how many quads
            to expect and a tuple of the bounding box of the grid
        quad_database_path (str): path to global database
        database_command_queue (queue): passed to _execute_sqlite so
            asyncronous writes can occur
        global_report_queue (queue): main work queue if not 'STOP' it can be
            a tuple indicating grid_id and bounding box, it can be a single
            element being a grid_id of one of its quads that that has been
            processed. When the count == 0 it will write the result to the
            database.

    Returns:
        None.
    """
    try:
        while True:
            payload = global_report_queue.get()
            LOGGER.debug(f'_global_grid_recorder payload {payload}')
            if payload == 'STOP':
                break
            if isinstance(payload, tuple):
                (grid_id, long_min, lat_min, long_max, lat_max,
                 quad_count) = payload
                global_processing_map[grid_id] = [
                    quad_count, (long_min, lat_min, long_max, lat_max)]
            else:
                grid_id = payload
                global_processing_map[grid_id][0] -= 1
                if global_processing_map[grid_id][0] < 0:
                    raise RuntimeError(
                        f'too many grid ids reported for {grid_id}')
                if global_processing_map[grid_id][0] == 0:
                    long_min, lat_min, long_max, lat_max = \
                        global_processing_map[grid_id][1]
                    _execute_sqlite(
                        '''
                        INSERT OR REPLACE INTO processed_grid_table
                            (grid_id, long_min, lat_min, long_max, lat_max,
                             status)
                        VALUES (?, ?, ?, ?, ?, ?);
                        ''', quad_database_path,
                        database_command_queue=database_command_queue,
                        execute='execute', argument_list=[
                            grid_id, long_min, lat_min, long_max, lat_max,
                            "complete"])
    except Exception:
        LOGGER.exception('something bad happened in _global_grid_recorder')


@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=5000)
def fetch_quad(
        session, quad_database_path, mosaic_id, quad_id, cache_dir,
        to_copy_queue, global_report_queue, grid_id):
    """Fetch quad from planet DB.

    Args:
        session (Session): session object to use for authentication
        quad_database_path (str): path to quad datbase
        mosaic_id (str): Planet mosaic ID to search for
        quad_id (str): Planet quad ID in the given mosaic to fetch
        cache_dir (str): path to directory to write temporary files in
        to_copy_queue (Queue): put 'OK' here when done with processing
        global_report_queue (Queue): used to report of the quad need not
            be downloaded.
        grid_id (str): unique id to report to global report queue if quad is
            predownloaded.
    Returns:
        None.
    """
    try:
        LOGGER.debug(f'fetching {quad_id}')
        count = _execute_sqlite(
            '''
            SELECT count(quad_id)
            FROM quad_cache_table
            WHERE quad_id=?;
            ''', quad_database_path, argument_list=[quad_id], fetch='one')
        LOGGER.debug(f'result of count query {count}')
        if count[0] > 0:
            LOGGER.debug('already fetched %s', quad_id)
            global_report_queue.put(grid_id)
            return

        get_quad_url = (
            f'https://api.planet.com/basemaps/v1/mosaics/'
            f'{mosaic_id}/quads/{quad_id}')
        quads_json = session.get(get_quad_url, timeout=REQUEST_TIMEOUT)
        download_url = (quads_json.json())['_links']['download']
        local_quad_path = os.path.join(cache_dir, '%s.tif' % quad_id)
        quad_uri = (
            'gs://natgeo-dams-data/cached-planet-quads/%s' %
            os.path.basename(local_quad_path))

        LOGGER.debug(f'download {download_url} to {local_quad_path}')
        ecoshard.download_url(download_url, local_quad_path)
        local_quad_info = pygeoprocessing.get_raster_info(local_quad_path)

        lng_lat_bb = pygeoprocessing.transform_bounding_box(
            local_quad_info['bounding_box'],
            local_quad_info['projection_wkt'],
            WGS84_WKT)

        sqlite_update_variables = []
        sqlite_update_variables.append(quad_id)
        sqlite_update_variables.extend(lng_lat_bb)
        sqlite_update_variables.append(  # file size in bytes
            pathlib.Path(local_quad_path).stat().st_size)
        sqlite_update_variables.append(quad_uri)

        LOGGER.debug(f'put {quad_uri} to copy from {sqlite_update_variables}')
        to_copy_queue.put(
            (local_quad_path, quad_uri, sqlite_update_variables))

    except Exception:
        LOGGER.exception('error on quad %s' % quad_id)
        raise


def _copy_quad_to_bucket_worker(
        grid_id, quad_database_path, database_command_queue, to_copy_queue,
        global_report_queue):
    """Copy downloaded quads to google bucket and note in  DB when done.

    Args:
        grid_id (str): unique ID of the global grid this quad is a part of
        quad_database_path (str): Path to database
        database_command_queue (queue): a queue to pass to _execute_sqlit
        to_copy_queue (queue): command queue that reports either quads to
            copy or to 'STOP'. Format of copy payload is
                (local_quad_path, quad_uri, sqlite_update_variables).
        global_report_queue (queue): send a 'grid_id' down this queue when
            quad is sucessfully copied to google bucket.

    Returns:
        None.
    """
    try:
        while True:
            try:
                payload = to_copy_queue.get()
                LOGGER.debug(f'copy quad to bucket payload {payload}')
                if payload == 'STOP':
                    break
                (local_quad_path, quad_uri, sqlite_update_variables) = payload
                subprocess.run(
                    '/usr/local/gcloud-sdk/google-cloud-sdk/bin/gsutil cp %s %s'
                    % (local_quad_path, quad_uri), shell=True, check=True)
            except subprocess.CalledProcessError:
                LOGGER.warning('file might already exist')

            global_report_queue.put(grid_id)

            try:
                os.remove(local_quad_path)
            except Exception:
                LOGGER.exception(f'could not remove {local_quad_path}')

            LOGGER.debug(
                'update sqlite table with these args: %s', sqlite_update_variables)
            _execute_sqlite(
                '''
                INSERT OR REPLACE INTO quad_cache_table
                    (quad_id, long_min, lat_min, long_max, lat_max, file_size,
                     gs_uri)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                ''', quad_database_path,
                database_command_queue=database_command_queue,
                execute='execute', argument_list=sqlite_update_variables)
        LOGGER.debug('all done with copy!')
    except Exception:
        LOGGER.exception('global exception on copy quad to bucket worker')


@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=5000)
def get_quad_ids(session, mosaic_id, min_x, min_y, max_x, max_y):
    """Search Planet for quads in a given bounding box.

    Args:
        session (Session): Session object to use for authentication
        mosaic_id (str): Planet mosaic to search in
        min_x, min_y, max_x, max_y (numeric): bounding box to search over

    Returns:
        None.
    """
    bb_query_url = (
        'https://api.planet.com/basemaps/v1/mosaics/'
        '%s/quads?bbox=%f,%f,%f,%f' % (
            mosaic_id, min_x, min_y, max_x, max_y))
    mosaics_response = session.get(bb_query_url, timeout=5.0)
    mosaics_json = mosaics_response.json()
    LOGGER.debug('%s: %s', mosaics_response, mosaics_json)
    quad_id_list = []
    while True:
        quad_id_list.extend(
            [item['id'] for item in mosaics_json['items']])
        if '_next' in mosaics_json['_links']:
            mosaics_json = session.get(
                mosaics_json['_links']['_next'], timeout=5.0).json()
        else:
            break
    return quad_id_list


def copy_from_gs(gs_uri, target_path):
    """Copy a GS objec to `target_path."""
    dirpath = os.path.dirname(target_path)
    try:
        os.makedirs(dirpath)
    except Exception:
        pass
    subprocess.run(
        '/usr/local/gcloud-sdk/google-cloud-sdk/bin/gsutil cp %s %s' %
        (gs_uri, target_path), shell=True)


def make_global_poly(vector_url):
    vector_path = os.path.join(CHURN_DIR, os.path.basename(vector_url))
    copy_from_gs(vector_url, vector_path)
    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    layer = vector.GetLayer()
    shapely_list = []
    for feature in layer:
        shapely_list.append(
            shapely.wkb.loads(feature.GetGeometryRef().ExportToWkb()))
        feature = None
    LOGGER.debug('global unary')
    global_shapely = shapely.ops.unary_union(shapely_list)
    return global_shapely


def main():
    """Entry point."""
    subprocess.run(
        '/usr/local/gcloud-sdk/google-cloud-sdk/bin/gcloud auth '
        f'activate-service-account --key-file natgeo-dams-fe937793db5d.json',
        shell=True, check=True)

    for dir_path in [WORKSPACE_DIR, ECOSHARD_DIR, CHURN_DIR, QUAD_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    # find the most recent mosaic we can use
    with open(PLANET_API_KEY_FILE, 'r') as planet_api_key_file:
        planet_api_key = planet_api_key_file.read().rstrip()

    LOGGER.debug(f'api key {planet_api_key}')
    session = requests.Session()
    session.auth = (planet_api_key, '')

    create_status_database(DATABASE_PATH)

    database_command_queue = multiprocessing.Queue()
    _database_manager_thread = threading.Thread(
        target=_database_manager,
        args=(DATABASE_PATH, database_command_queue))
    _database_manager_thread.start()

    global_report_queue = multiprocessing.Queue()
    work_process_list = []
    work_queue = multiprocessing.Queue(N_WORKERS)
    for worker_id in range(N_WORKERS):
        work_process = multiprocessing.Process(
            target=fetch_quad_worker,
            args=(work_queue, planet_api_key, DATABASE_PATH, QUAD_DIR,
                  database_command_queue, global_report_queue))
        work_process.daemon = True
        work_process.start()
        work_process_list.append(work_process)

    global_processing_map = {}
    _global_grid_recorder_thread = threading.Thread(
        target=_global_grid_recorder,
        args=(
            global_processing_map, DATABASE_PATH, database_command_queue,
            global_report_queue))
    _global_grid_recorder_thread.daemon = True
    _global_grid_recorder_thread.start()

    # fetch all the quad ids that have already been fetched
    quad_id_query = _execute_sqlite(
        """
        SELECT quad_id
        FROM quad_cache_table
        """, DATABASE_PATH, fetch='all')
    quad_id_set = set(x[0] for x in quad_id_query)

    # fetch all the grids (grids contain quads) that have already been fetched
    processed_grid_id_query = _execute_sqlite(
        """
        SELECT grid_id
        FROM processed_grid_table
        """, DATABASE_PATH, fetch='all')
    processed_grid_set = set(x[0] for x in processed_grid_id_query)

    LOGGER.info('prep the country lookup structure')
    avoid_countries = set(['ATA', 'GRL'])
    country_vector_path = 'countries_iso3_md5_6fb2431e911401992e6e56ddf0a9bcda.gpkg'
    country_vector = gdal.OpenEx(country_vector_path, gdal.OF_VECTOR)
    country_layer = country_vector.GetLayer()
    all_country_geom_list = []
    for country_feature in country_layer:
        country_geom = country_feature.GetGeometryRef()
        country_shp = shapely.wkb.loads(country_geom.ExportToWkb())
        country_iso = country_feature.GetField('iso3')
        if country_iso not in avoid_countries:
            all_country_geom_list.append(country_shp)
    LOGGER.info('prep the geometry for fast intersection')
    all_country_union = shapely.ops.cascaded_union(all_country_geom_list)
    all_country_prep = shapely.prepared.prep(all_country_union)

    for lat in range(71, -90, -1):
        for lng in range(-180, 180):
            grid_id = (lat+90)*360+lng+180
            box = shapely.geometry.box(lng, lat-1, lng+1, lat)
            # make sure we intersect a country
            if not all_country_prep.intersects(box):
                continue
            # check to see if we've processed this grid before, if so, skip
            if grid_id in processed_grid_set:
                continue
            # get planet quad lists for that grid ID
            quad_id_list = get_quad_ids(
                session, MOSAIC_ID, lng, lat, lng+1, lat+1)
            if not quad_id_list:
                LOGGER.debug(f'no quads found at {lat}N {lng}W ')
                continue
            # remove any quads we've already processed
            for quad_id in list(quad_id_list):
                if quad_id in quad_id_set:
                    quad_id_list.remove(quad_id)
                    continue
                quad_id_set.add(quad_id)

            LOGGER.debug(
                'processing these quads %d %d %s', lat, lng, str(quad_id_list))
            # work queue will take an entire grid and quad list
            work_queue.put(
                (MOSAIC_ID, grid_id, lng, lat, lng+1, lat+1, quad_id_list))
            break
        break

    work_queue.put('STOP')
    for worker_process in work_process_list:
        worker_process.join()
    LOGGER.debug('ALL DONE!')
    LOGGER.debug("JOINING database manager thread, this might not stop")
    _database_manager_thread.join()
    _global_grid_recorder.join()
    LOGGER.debug('IT DID STOP!')


if __name__ == '__main__':
    main()
