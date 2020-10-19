"""Tracer code to set up training pipeline."""
import os
import logging
import pathlib
import sqlite3
import subprocess
import sys

import ecoshard
import requests
import retrying
import taskgraph

WORKSPACE_DIR = 'quad_fetcher_workspace'
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
QUAD_DIR = os.path.join(CHURN_DIR, 'quads')

COUNTRY_BORDER_VECTOR_URI = (
    'gs://natgeo-dams-data/ecoshards/'
    'countries_iso3_md5_6fb2431e911401992e6e56ddf0a9bcda.gpkg')

# This is the Planet Mosaic ID for global_quarterly_2019q2_mosaic
MOSAIC_ID = '4ce5863a-fb3f-4cad-a899-b8c053af1858'
# This was pre-calculated

PLANET_API_KEY_FILE = 'planet_api_key.txt'
REQUEST_TIMEOUT = 1.5

logging.basicConfig(
        level=logging.DEBUG,
        format=(
            '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
            '%(name)s [%(funcName)s:%(lineno)d] %(message)s'),
        stream=sys.stdout)
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.FileHandler('log.txt'))
#logging.getLogger('taskgraph').setLevel(logging.INFO)


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
        CREATE TABLE bounding_box_to_mosaic (
            record_id INT NOT NULL PRIMARY KEY,
            bounding_box BLOB NOT NULL,
            mosaic_id TEXT NOT NULL);
        """)
    if os.path.exists(database_path):
        os.remove(database_path)
    connection = sqlite3.connect(database_path)
    connection.execute(create_database_sql)
    connection.commit()
    connection.close()


@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=5000)
def fetch_quad(planet_api_key, mosaic_id, quad_id, target_quad_path):
    try:
        session = requests.Session()
        session.auth = (planet_api_key, '')
        LOGGER.debug('fetch get quad')
        get_quad_url = (
            f'https://api.planet.com/basemaps/v1/mosaics/'
            f'{mosaic_id}/quads/{quad_id}')
        quads_json = session.get(get_quad_url, timeout=REQUEST_TIMEOUT)
        download_url = (quads_json.json())['_links']['download']
        ecoshard.download_url(download_url, target_quad_path)
        quad_uri = (
            'gs://natgeo-dams-data/known-dam-quads/%s' %
            os.path.basename(target_quad_path))
        subprocess.run(
            './google-cloud-sdk/bin/gsutil cp %s %s' % (target_quad_path, quad_uri),
            shell=True, check=True)
        os.remove(target_quad_path)
        insert_quad_url_into = (
            "INSERT INTO "
            "quad_id_to_uri (quad_id, quad_uri) "
            "VALUES (?, ?);")
        _execute_sqlite(
            insert_quad_url_into, quad_database_path,
            mode='modify', execute='execute',
            argument_list=[quad_id, quad_uri])
        return True
    except Exception:
        LOGGER.exception('error on quad %s' % quad_id)
        raise


def copy_from_gs(gs_uri, target_path):
    """Copy a GS objec to `target_path."""
    dirpath = os.path.dirname(target_path)
    try:
        os.makedirs(dirpath)
    except Exception:
        pass
    subprocess.run(
        #'/usr/local/gcloud-sdk/google-cloud-sdk/bin/gsutil cp %s %s' %
        'gsutil cp %s %s' %
        (gs_uri, target_path), shell=True)


if __name__ == '__main__':
    for dir_path in [WORKSPACE_DIR, CHURN_DIR, QUAD_DIR]:
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

    country_borders_vector_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(COUNTRY_BORDER_VECTOR_URI))
    country_borders_dl_task = task_graph.add_task(
        func=copy_from_gs,
        args=(
            COUNTRY_BORDER_VECTOR_URI,
            country_borders_vector_path),
        task_name='download country borders vector',
        target_path_list=[country_borders_vector_path])

    quads_to_download_query = (
        """
        SELECT bounding_box_to_mosaic.quad_id
        FROM bounding_box_to_mosaic
        LEFT JOIN quad_id_to_uri ON
            bounding_box_to_mosaic.quad_id = quad_id_to_uri.quad_id
        WHERE quad_id_to_uri.quad_id IS NULL;
        """)

    result = _execute_sqlite(
        quads_to_download_query, quad_database_path,
        execute='execute', argument_list=[], fetch='all')

    for ((quad_id,)) in result:
        quad_path = os.path.join(
            QUAD_DIR, '%s_%s.tif' % (MOSAIC_ID, quad_id))
        fetch_quad_task = task_graph.add_task(
            func=fetch_quad,
            args=(planet_api_key, MOSAIC_ID, quad_id, quad_path),
            task_name='fetch %s_%s' % (MOSAIC_ID, quad_id))
        if not fetch_quad_task.get():
            raise RuntimeError('fetch %s failed' % quad_path)

    LOGGER.debug(fetch_quad_task.get())
    LOGGER.debug('closing')
    task_graph.close()
    task_graph.join()
