"""Tracer code to set up training pipeline."""
import os
import logging
import pathlib
import sqlite3
import subprocess

from osgeo import gdal
import requests
import retrying
import rtree
import shapely.geometry
import shapely.ops
import shapely.prepared
import shapely.wkb
import taskgraph

WORKSPACE_DIR = 'planet_cell_to_grid_cache_workspace'
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')
DATABASE_PATH = os.path.join(WORKSPACE_DIR, 'planet_cell_to_grid.db')

COUNTRY_BORDER_VECTOR_URI = (
    'gs://natgeo-dams-data/ecoshards/'
    'countries_iso3_md5_6fb2431e911401992e6e56ddf0a9bcda.gpkg')

ISO_CODES_TO_SKIP = ['ATA']
# This is the Planet Mosaic ID for global_quarterly_2019q2_mosaic
MOSAIC_ID = '4ce5863a-fb3f-4cad-a899-b8c053af1858'
PLANET_API_KEY_FILE = 'planet_api_key.txt'
REQUEST_TIMEOUT = 1.5

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'),
    filename='quad_cache_log.txt')
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
        CREATE TABLE grid_id_to_quad_id (
            grid_id INTEGER NOT NULL PRIMARY KEY,
            quad_id_list TEXT NOT NULL,
            country_iso3_list TEXT NOT NULL,
            lng_min REAL NOT NULL,
            lat_min REAL NOT NULL,
            lng_max REAL NOT NULL,
            lat_max REAL NOT NULL
            );
        """)
    if os.path.exists(database_path):
        os.remove(database_path)
    connection = sqlite3.connect(database_path)
    connection.executescript(create_database_sql)
    connection.commit()
    connection.close()


def copy_from_gs(gs_uri, target_path):
    """Copy a GS objec to `target_path."""
    dirpath = os.path.dirname(target_path)
    try:
        os.makedirs(dirpath)
    except Exception:
        pass
    subprocess.run(
        'gsutil cp %s %s' % (gs_uri, target_path), shell=True)


@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=5000)
def get_quad_ids(session, mosaic_id, min_x, min_y, max_x, max_y):
    bb_query_url = (
        'https://api.planet.com/basemaps/v1/mosaics/'
        '%s/quads?bbox=%f,%f,%f,%f' % (
            mosaic_id, min_x, min_y, max_x, max_y))
    mosaics_response = session.get(bb_query_url, timeout=5.0)
    mosaics_json = mosaics_response.json()
    LOGGER.debug('quad response %s: %s', mosaics_response, mosaics_json)
    quad_id_list = []
    while True:
        quad_id_list.extend(
            [item['id'] for item in mosaics_json['items']])
        if '_next' in mosaics_json['_links']:
            LOGGER.debug('_next in %s')
            mosaics_json = session.get(
                mosaics_json['_links']['_next'], timeout=5.0).json()
        else:
            break
    return quad_id_list


def get_country_intersection_list(grid_box, country_vector_path):
    """Query index and return string list of intersecting countries.

    Parameters:
        grid_box (list): list of lng_min, lat_min, lng_max, lat_max
        country_vector_path (str): path to country index vector with 'iso3'
            fields.

    Returns:
        list of string country ISO3 codes.

    """
    if not hasattr(get_country_intersection_list, 'country_index'):
        # TODO: make country index
        country_vector = gdal.OpenEx(country_vector_path, gdal.OF_VECTOR)
        country_layer = country_vector.GetLayer()

        country_index = rtree.index.Index()
        country_index.country_geom_list = []
        country_index.country_iso3_list = []
        for country_feature in country_layer:
            country_geom = country_feature.GetGeometryRef()
            country_shapely = shapely.wkb.loads(country_geom.ExportToWkb())
            country_geom = None
            country_index.insert(
                len(country_index.country_geom_list), country_shapely.bounds)
            country_index.country_geom_list.append(country_shapely)
            country_index.country_iso3_list.append(
                country_feature.GetField('iso3'))
        get_country_intersection_list.country_index = country_index

    country_index = get_country_intersection_list.country_index
    intersecting_country_list = []
    for country_id in country_index.intersection(grid_box.bounds):
        if country_index.country_geom_list[country_id].intersects(grid_box):
            intersecting_country_list.append(
                country_index.country_iso3_list[country_id])
    return intersecting_country_list


def main():
    for dir_path in [WORKSPACE_DIR, ECOSHARD_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    with open(PLANET_API_KEY_FILE, 'r') as planet_api_key_file:
        planet_api_key = planet_api_key_file.read().rstrip()

    session = requests.Session()
    session.auth = (planet_api_key, '')

    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1, 5.0)

    task_graph.add_task(
        func=create_status_database,
        args=(DATABASE_PATH,),
        target_path_list=[DATABASE_PATH],
        task_name='create database')

    country_borders_vector_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(COUNTRY_BORDER_VECTOR_URI))
    country_borders_dl_task = task_graph.add_task(
        func=copy_from_gs,
        args=(
            COUNTRY_BORDER_VECTOR_URI,
            country_borders_vector_path),
        task_name='download country borders vector',
        target_path_list=[country_borders_vector_path])
    country_borders_dl_task.join()
    LOGGER.debug('start quad search')

    grid_insert_args = []
    grid_id = 0
    for lat_max in range(-60, 60):
        LOGGER.debug(lat_max)
        for lng_min in range(-180, 180):
            grid_box = shapely.geometry.box(
                lng_min, lat_max-1, lng_min+1, lat_max)
            intersecting_country_list = \
                get_country_intersection_list(
                    grid_box, country_borders_vector_path)
            if not intersecting_country_list or \
                    ISO_CODES_TO_SKIP in intersecting_country_list:
                continue

            grid_exists = _execute_sqlite(
                """
                SELECT count(*)
                FROM grid_id_to_quad_id
                WHERE grid_id=?
                """, DATABASE_PATH,
                argument_list=[grid_id], mode='read_only', execute='execute',
                fetch='one')[0]

            # if grid not already in the database, grid_exists will be 0
            if not grid_exists:
                # query planet API for these bounding boxes
                quad_id_list = get_quad_ids(
                    session, MOSAIC_ID, lng_min, lat_max-1, lng_min+1, lat_max)
                LOGGER.debug(quad_id_list)
            #   TODO: insert into DB

            """
            CREATE TABLE grid_id_to_quad_id (
                grid_id INTEGER NOT NULL PRIMARY KEY,
                quad_id_list TEXT NOT NULL,
                country_iso3_list TEXT NOT NULL,
                lng_min REAL NOT NULL,
                lat_min REAL NOT NULL,
                lng_max REAL NOT NULL,
                lat_max REAL NOT NULL
                );
            """

            grid_insert_args.append((
                grid_id, lng_min, lat_max-1, lng_min+1, lat_max,
                ','.join(intersecting_country_list), 0))
            grid_id += 1


if __name__ == '__main__':
    main()
