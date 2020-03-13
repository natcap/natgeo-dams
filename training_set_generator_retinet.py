"""Tracer code to set up training pipeline."""
import os
import logging
import pathlib
import pickle
import sqlite3
import subprocess
import sys

import ecoshard
import requests
import retrying
import taskgraph

WORKSPACE_DIR = 'training_set_workspace'
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')
ANNOTATIONS_CSV_PATH = os.path.join(WORKSPACE_DIR, 'annotations.csv')
CLASSES_CSV_PATH = os.path.join(WORKSPACE_DIR, 'classes.csv')

PLANET_QUAD_DAMS_DATABASE_URI = (
    'gs://natgeo-dams-data/databases/'
    'quad_database_md5_12866cf27da2575f33652d197beb05d3.db')

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


def make_training_data(
        dams_database_path, annotations_csv_path, classes_csv_path):
    """Make training data by fetching imagery and building CSVs.

    Parameters:
        dams_database_path (str): path to database containing a
            "bounding_box_to_mosaic" table.
        annotations_csv_path (str): path to a csv containing annotations.
            Each line is of the form: path/to/image.jpg,x1,y1,x2,y2,class_name
        classes_csv_path (str): path to csv containing classes definitions
            each line containing the row [class name],[id]. Might only be
            "dam,0"

    Returns:
        None

    """
    result = _execute_sqlite(
        '''
        SELECT bounding_box
        FROM bounding_box_to_mosaic
        WHERE quad_id='757-890';
        ''', dams_database_path, argument_list=[], fetch='all')
    for (bounding_box_pickled,) in result:
        bounding_box = pickle.loads(bounding_box_pickled)
        LOGGER.debug(bounding_box)


def copy_from_gs(gs_uri, target_path):
    """Copy a GS objec to `target_path."""
    dirpath = os.path.dirname(target_path)
    try:
        os.makedirs(dirpath)
    except Exception:
        pass
    subprocess.run(
        'gsutil cp %s %s' % (gs_uri, target_path), shell=True, check=True)


def main():
    """Entry point."""
    for dir_path in [WORKSPACE_DIR, ECOSHARD_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1)

    PLANET_QUAD_DAMS_DATABASE_URI

    planet_quad_dams_database_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(PLANET_QUAD_DAMS_DATABASE_URI))
    task_graph.add_task(
        func=copy_from_gs,
        args=(
            PLANET_QUAD_DAMS_DATABASE_URI,
            planet_quad_dams_database_path),
        task_name='download planet quad db',
        target_path_list=[planet_quad_dams_database_path])
    task_graph.join()

    make_training_data(
        planet_quad_dams_database_path, ANNOTATIONS_CSV_PATH, CLASSES_CSV_PATH)


if __name__ == '__main__':
    main()
