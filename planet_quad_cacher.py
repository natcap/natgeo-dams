"""Tracer code to set up training pipeline."""
import os
import logging
import pathlib
import sqlite3
import subprocess
import sys

from osgeo import gdal
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

COUNTRY_BORDER_VECTOR_URI = (
    'gs://natgeo-dams-data/ecoshards/'
    'countries_iso3_md5_6fb2431e911401992e6e56ddf0a9bcda.gpkg')

# This is the Planet Mosaic ID for global_quarterly_2019q2_mosaic
MOSAIC_ID = '4ce5863a-fb3f-4cad-a899-b8c053af1858'
PLANET_API_KEY_FILE = 'planet_api_key.txt'
REQUEST_TIMEOUT = 1.5

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=5000)
def _execute_sqlite(
        sqlite_command, database_path, argument_list=None,
        mode='read_only', execute='execute', fetch=None):
    """Execute SQLite command and attempt retries on a failure.

    Parameters:
        sqlite_command (str): a well formatted SQLite command.
        database_path (str): path to the SQLite database to operate on.
        argument_list (list): `execute == 'execute` then this list is passed to
            the internal sqlite3 `execute` call.
        mode (str): must be either 'read_only' or 'modify'.
        execute (str): must be either 'execute', 'many', or 'script'.
        fetch (str): if not `None` can be either 'all' or 'one'.
            If not None the result of a fetch will be returned by this
            function.

    Returns:
        result of fetch if `fetch` is not None.

    """
    cursor = None
    connection = None
    try:
        if mode == 'read_only':
            ro_uri = r'%s?mode=ro' % pathlib.Path(
                os.path.abspath(database_path)).as_uri()
            LOGGER.debug(
                '%s exists: %s', ro_uri, os.path.exists(os.path.abspath(
                    database_path)))
            connection = sqlite3.connect(ro_uri, uri=True)
        elif mode == 'modify':
            connection = sqlite3.connect(database_path)
        else:
            raise ValueError('Unknown mode: %s' % mode)

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

    Parameters:
        database_path (str): path to database to create.

    Returns:
        None.

    """
    LOGGER.debug('launching create_status_database')
    create_database_sql = (
        """
        CREATE TABLE quad_cache_table (
            quad_id TEXT NOT NULL PRIMARY KEY,
            long_min FLOAT NOT NULL,
            lat_min FLOAT NOT NULL,
            long_max FLOAT NOT NULL,
            lat_max FLOAT NOT NULL,
            file_size INTEGER NOT NULL,
            gs_uri TEXT NOT NULL
            );

        CREATE INDEX long_min_index ON quad_cache_table (long_min);
        CREATE INDEX lat_min_index ON quad_cache_table (lat_min);
        CREATE INDEX long_max_index ON quad_cache_table (long_max);
        CREATE INDEX lat_max_index ON quad_cache_table (lat_max);
        """)
    if os.path.exists(database_path):
        os.remove(database_path)
    connection = sqlite3.connect(database_path)
    connection.executescript(create_database_sql)
    connection.commit()
    connection.close()


@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=5000)
def fetch_quad(
        session, quad_database_path, planet_api_key, mosaic_id, quad_id,
        cache_dir):
    try:
        count = _execute_sqlite(
            '''
            SELECT count(*)
            FROM quad_cache_table
            WHERE quad_id=?;
            ''', quad_database_path, argument_list=[quad_id], fetch='one')
        if count[0] > 0:
            LOGGER.debug('already fetched %s', quad_id)
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

        ecoshard.download_url(download_url, local_quad_path)
        local_quad_info = pygeoprocessing.get_raster_info(local_quad_path)

        sqlite_update_variables = []
        sqlite_update_variables.append(quad_id)
        sqlite_update_variables.extend(local_quad_info['bounding_box'])
        sqlite_update_variables.append(  # file size in bytes
            pathlib.Path(local_quad_path).stat().st_size)
        sqlite_update_variables.append(quad_uri)

        try:
            ls_result = subprocess.run(
                'gsutil ls %s' % quad_uri, capture_output=True, shell=True,
                check=True)
            LOGGER.debug(ls_result)
            subprocess.run(
                '/usr/local/gcloud-sdk/google-cloud-sdk/bin/gsutil mv %s %s'
                % (local_quad_path, quad_uri), shell=True, check=True)
        except subprocess.CalledProcessError:
            LOGGER.exception('file might already exist')

        LOGGER.debug(
            'update sqlite table with these args: %s', sqlite_update_variables)
        _execute_sqlite(
            '''
            INSERT INTO quad_cache_table
                (quad_id, long_min, lat_min, long_max, lat_max, file_size,
                 gs_uri)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            ''', quad_database_path,
            mode='modify', execute='execute',
            argument_list=sqlite_update_variables)

        return True
    except Exception:
        LOGGER.exception('error on quad %s' % quad_id)
        raise


@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=5000)
def get_quad_ids(session, mosaic_id, min_x, min_y, max_x, max_y):
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


if __name__ == '__main__':
    for dir_path in [WORKSPACE_DIR, ECOSHARD_DIR, CHURN_DIR, QUAD_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    # find the most recent mosaic we can use
    with open(PLANET_API_KEY_FILE, 'r') as planet_api_key_file:
        planet_api_key = planet_api_key_file.read().rstrip()

    session = requests.Session()
    session.auth = (planet_api_key, '')

    task_graph = taskgraph.TaskGraph(CHURN_DIR, -1, 5.0)

    task_graph.add_task(
        func=create_status_database,
        args=(DATABASE_PATH,),
        target_path_list=[DATABASE_PATH],
        task_name='create database')

    global_poly_task = task_graph.add_task(
        func=make_global_poly,
        args=(COUNTRY_BORDER_VECTOR_URI,),
        task_name='make global poly')
    task_graph.join()

    LOGGER.debug('load countries to shapely')

    LOGGER.debug('global prep')
    global_shapely_prep = shapely.prepared.prep(global_poly_task.get())
    LOGGER.debug('start quad search')

    for lat in range(-60, 60):
        for lng in range(-180, 180):
            query_box = shapely.geometry.box(lng, lat, lng+1, lat+1)
            if not global_shapely_prep.intersects(query_box):
                continue
            quad_id_list = get_quad_ids(
                session, MOSAIC_ID, lng, lat, lng+1, lat+1)
            LOGGER.debug('%d %d %s', lat, lng, str(quad_id_list))
            if not quad_id_list:
                continue
            for quad_id in quad_id_list:
                fetch_quad(
                    session, DATABASE_PATH, planet_api_key, MOSAIC_ID, quad_id,
                    QUAD_DIR)
                break
            break
        break

    task_graph.close()
    task_graph.join()
    LOGGER.debug('ALL DONE!')