"""Tracer code to set up training pipeline."""
import os
import logging
import pathlib
import pickle
import sqlite3
import sys

import ecoshard
import requests
import retrying
import taskgraph

WORKSPACE_DIR = 'training_tracer_workspace'
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')
STATUS_DATABASE_PATH = os.path.join(WORKSPACE_DIR, 'status_database.db')

KNOWN_DAMS_DATABASE_URL = (
    'https://storage.googleapis.com/natcap-natgeo-dam-ecoshards/'
    'find-the-dams_report_md5_e12d71eb5026751217bd286e9b4c2afe.db')

PLANET_API_KEY_FILE = 'planet_api_key.txt'
ACTIVE_MOSAIC_JSON_PATH = os.path.join(WORKSPACE_DIR, 'active_mosaic.json')

REQUEST_TIMEOUT = 1.5

logging.basicConfig(
        level=logging.DEBUG,
        format=(
            '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
            '%(name)s [%(funcName)s:%(lineno)d] %(message)s'),
        stream=sys.stdout)
LOGGER = logging.getLogger(__name__)
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


def get_planet_mosaic(mosaic_name):
    """Return the planet mosaic to use.

    Parameters:
        mosaic_name (str): mosaic name from Planet referece to use.


    Returns:
        dictionary of mosaic object described in
        https://developers.planet.com/docs/basemaps/reference/#tag/Basemaps-and-Mosaics
    """
    LOGGER.debug('get_planet_mosaic')
    mosaics_json = session.get(
        'https://api.planet.com/basemaps/v1/mosaics?name__is=%s' % mosaic_name,
        timeout=REQUEST_TIMEOUT)
    return (mosaics_json.json())['mosaics'][0]


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


if __name__ == '__main__':
    for dir_path in [WORKSPACE_DIR, CHURN_DIR, ECOSHARD_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    # find the most recent mosaic we can use
    with open(PLANET_API_KEY_FILE, 'r') as planet_api_key_file:
        planet_api_key = planet_api_key_file.read().rstrip()

    session = requests.Session()
    session.auth = (planet_api_key, '')

    task_graph = taskgraph.TaskGraph(CHURN_DIR, 0, 5.0)

    known_dams_database_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(KNOWN_DAMS_DATABASE_URL))
    download_database_task = task_graph.add_task(
        func=ecoshard.download_url,
        args=(KNOWN_DAMS_DATABASE_URL, known_dams_database_path),
        target_path_list=[known_dams_database_path],
        task_name='download %s' % KNOWN_DAMS_DATABASE_URL)

    create_status_database_task = task_graph.add_task(
        func=create_status_database,
        args=(STATUS_DATABASE_PATH,),
        hash_target_files=False,
        target_path_list=[STATUS_DATABASE_PATH],
        task_name='create status database')

    # _execute_sqlite()
    get_mosaic_task = task_graph.add_task(
        func=get_planet_mosaic,
        args=('global_quarterly_2019q2_mosaic',),
        task_name='get planet mosaic for %s' %
        'global_quarterly_2019q2_mosaic')
    get_mosaic_task.join()
    active_mosaic = get_mosaic_task.get()
    LOGGER.debug('active_mosaic: %s', active_mosaic)

    known_dam_query = (
        "SELECT lng_min, lat_min, lng_max, lat_max "
        "FROM identified_dams "
        "WHERE pre_known=1 AND dam_description NOT LIKE '%South Africa%' "
        "ORDER BY dam_description;")

    result = _execute_sqlite(
        known_dam_query, known_dams_database_path,
        execute='execute', argument_list=[], fetch='all')

    argument_list = []
    for index, (lng_min, lat_min, lng_max, lat_max) in enumerate(result):
        argument_list.append(
            (index, pickle.dumps((lng_min, lat_min, lng_max, lat_max)), 'X'))
    # add the bounding boxes for identified dams into the job status database
    insert_identified_dams_query = (
        "INSERT OR IGNORE INTO "
        "bounding_box_to_mosaic (record_id, bounding_box, mosaic_id) "
        "VALUES (?, ?, ?);")
    create_status_database_task.join()
    _execute_sqlite(
        insert_identified_dams_query, STATUS_DATABASE_PATH,
        mode='modify', execute='many', argument_list=argument_list)

    bbs_to_find = _execute_sqlite(
        'SELECT record_id, bounding_box '
        'FROM bounding_box_to_mosaic '
        'WHERE mosaic_id = \'X\'', STATUS_DATABASE_PATH,
        mode='read_only', execute='execute', argument_list=[], fetch='all')

    for record_id, bounding_box_pickle in bbs_to_find:
        LOGGER.debug(record_id)

    # for r in result:
    #     LOGGER.debug(r)

    # quads_json = session.get(
    #     'https://api.planet.com/basemaps/v1/mosaics/{%s}/'
    #     'quads?bbox=lx,ly,ux,uy' % (
    #         'foo,',
    #         active_mosaic['id']), timeout=REQUEST_TIMEOUT)

    #LOGGER.debug(quads_json.json())

    task_graph.close()
    task_graph.join()
