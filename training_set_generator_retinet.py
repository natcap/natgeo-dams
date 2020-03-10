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

WORKSPACE_DIR = 'training_set_workspace'
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')
ANNOTATIONS_CSV_PATH = os.path.join(WORKSPACE_DIR, 'annotations.csv')
CLASSES_CSV_PATH = os.path.join(WORKSPACE_DIR, 'classes.csv')

KNOWN_DAMS_DATABASE_URL = (
    'https://storage.googleapis.com/natcap-natgeo-dam-ecoshards/'
    'find-the-dams_report_md5_e12d71eb5026751217bd286e9b4c2afe.db')


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


def main():
    """Entry point."""
    for dir_path in [WORKSPACE_DIR, ECOSHARD_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1)

    known_dams_db_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(KNOWN_DAMS_DATABASE_URL))
    task_graph.add_task(
        func=ecoshard.download_url,
        args=(KNOWN_DAMS_DATABASE_URL, known_dams_db_path),
        target_path_list=[known_dams_db_path])
    pass


if __name__ == '__main__':
    main()
