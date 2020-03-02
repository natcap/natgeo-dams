"""Tracer code to set up training pipeline."""
import os
import logging
import sys

import ecoshard
import requests
from taskgraph.Task import _execute_sqlite
import taskgraph

WORKSPACE_DIR = 'training_tracer_workspace'
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')

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

    task_graph = taskgraph.TaskGraph(CHURN_DIR, -1, 5.0)

    known_dams_database_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(KNOWN_DAMS_DATABASE_URL))
    download_database_task = task_graph.add_task(
        func=ecoshard.download_url,
        args=(KNOWN_DAMS_DATABASE_URL, known_dams_database_path),
        target_path_list=[known_dams_database_path],
        task_name='download %s' % KNOWN_DAMS_DATABASE_URL)

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
    for r in result:
        LOGGER.debug(r)

    # quads_json = session.get(
    #     'https://api.planet.com/basemaps/v1/mosaics/{%s}/'
    #     'quads?bbox=lx,ly,ux,uy' % (
    #         'foo,',
    #         active_mosaic['id']), timeout=REQUEST_TIMEOUT)

    #LOGGER.debug(quads_json.json())

    task_graph.close()
    task_graph.join()
