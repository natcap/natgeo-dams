"""Tracer code to set up training pipeline."""
import os
import logging
import multiprocessing
import pathlib
import pickle
import sqlite3
import subprocess
import sys

from osgeo import osr
from osgeo import gdal
import numpy
import pygeoprocessing
import png
import retrying
import rtree
import taskgraph

"""
git pull && docker build dockerfile-dir -f dockerfile-dir/docker-cpu -t natcap/dam-inference-server-cpu:0.0.1 && docker run -it --rm -v `pwd`:/usr/local/natgeo_dams natcap/dam-inference-server-cpu:0.0.1 python "./training_set_generator_retinet.py"
"""


WORKSPACE_DIR = 'training_set_workspace'
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
ANNOTATIONS_CSV_PATH = os.path.join('.', 'annotations.csv')
CLASSES_CSV_PATH = os.path.join('.', 'classes.csv')
TRAINING_IMAGERY_DIR = os.path.join(WORKSPACE_DIR, 'training_imagery')
PLANET_QUAD_DAMS_DATABASE_URI = (
    'gs://natgeo-dams-data/databases/'
    'quad_database_md5_12866cf27da2575f33652d197beb05d3.db')
STATUS_DATABASE_PATH = os.path.join(CHURN_DIR, 'work_status.db')

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'),
    filename='log.txt')
LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)

TRAINING_IMAGE_DIMS = (419, 419)


def create_status_database(quads_database_path, target_status_database_path):
    """Create a runtime status database if it doesn't exist.

    Parameters:
        quads_database_path (str): path to existing database of quads.
        target_status_database_path (str): path to database to create.

    Returns:
        None.

    """
    LOGGER.debug('launching create_status_database')
    bounding_box_quad_uri_list = _execute_sqlite(
        '''
        SELECT
            quad_id, quad_id_to_uri.quad_uri, bounding_box
        FROM bounding_box_to_mosaic
        INNER JOIN quad_id_to_uri ON
            bounding_box_to_mosaic.quad_id = quad_id_to_uri.quad_id
        ''', quads_database_path, argument_list=[], fetch='all')

    quad_id_set = set([[x[0]] for x in bounding_box_quad_uri_list])

    # processed quads table
    # annotations
    create_database_sql = (
        """
        CREATE TABLE quad_bounding_box_uri_table (
            quad_id TEXT NOT NULL,
            quad_uri TEXT NOT NULL,
            bounding_box BLOB NOT NULL);

        CREATE TABLE quad_processing_status (
            quad_id TEXT NOT NULL,
            processed INT NOT NULL);
        """)
    if os.path.exists(target_status_database_path):
        os.remove(target_status_database_path)
    connection = sqlite3.connect(target_status_database_path)
    connection.executescript(create_database_sql)
    connection.commit()
    connection.close()

    _execute_sqlite(
        "INSERT INTO "
        "quad_bounding_box_uri_table (quad_id, quad_uri, bounding_box) "
        "VALUES (?, ?, ?);",
        target_status_database_path,
        argument_list=bounding_box_quad_uri_list, mode='modify',
        execute='many')

    _execute_sqlite(
        "INSERT INTO "
        "quad_processing_status (quad_id, processed) "
        "VALUES (?, 0);",
        target_status_database_path,
        argument_list=quad_id_set, mode='modify', execute='many')


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
        task_graph, dams_database_path, imagery_dir, annotations_csv_path,
        classes_csv_path):
    """Make training data by fetching imagery and building CSVs.

    Parameters:
        task_graph (taskgraph.Taskgraph): TaskGraph object to help with
            scheduling downloads.
        dams_database_path (str): path to database containing a
            "bounding_box_to_mosaic" table.
        imagery_dir (str): path to directory to store images.
        annotations_csv_path (str): path to a csv containing annotations.
            Each line is of the form: path/to/image.jpg,x1,y1,x2,y2,class_name
        classes_csv_path (str): path to csv containing classes definitions
            each line containing the row [class name],[id]. Might only be
            "dam,0"

    Returns:
        None

    """

    # get all the quad_ids

    quad_id_uris_to_process = _execute_sqlite(
        '''
        SELECT
            quad_id, quad_bounding_box_uri_table.quad_uri
        FROM quad_processing_status
        INNER JOIN
            quad_processing_status.quad_id=quad_bounding_box_uri_table.quad_id
        WHERE processed=0
        ''', dams_database_path, argument_list=[], fetch='all')

    for quad_id, quad_uri in quad_id_uris_to_process:
        _ = task_graph.add_task(
            func=process_quad,
            args=(quad_uri, quad_id, dams_database_path),
            task_name='process quad %s' % quad_id)

    task_graph.join()

    with open(classes_csv_path, 'w') as classes_csv_file:
        classes_csv_file.write('dam,0\n')
    annotations_csv_file = open(annotations_csv_path, 'w')



    # for each quad id
    #   download the quad
    #   get all the bounding boxes, transform them, make them local coordinates
    #   put in r-tree
    #   cut quad into sizes that are the same size as "not a dam"
    #   for each quad find all the bounding boxes that intersect it
    #       if there are some, create a png, and annotations in db to support it
    #       note the local png will need local coordinates
    #   update the database that the quad is annodated
    #   delete the quad

    '''
        SELECT
            quad_id, quad_id_to_uri.quad_uri, bounding_box
        FROM bounding_box_to_mosaic
        INNER JOIN quad_id_to_uri ON
            bounding_box_to_mosaic.quad_id = quad_id_to_uri.quad_id
    '''
    """
        CREATE TABLE quad_bounding_box_uri_table (
            quad_id TEXT NOT NULL,
            quad_uri TEXT NOT NULL,
            bounding_box BLOB NOT NULL);

        CREATE TABLE quad_processing_status (
            quad_id TEXT NOT NULL,
            processed INT NOT NULL);
    """


def process_quad(quad_uri, quad_id, dams_database_path):
    """Process quad into bounding box annotated chunks.

    Parameters:
        quad_uri (str): gs:// path to quad to download.
        quad_id (str): ID in the database so work can be updated.
        dams_database_path (str): path to the database that can be
            updated to include the processing state complete and the
            quad processed.

    Returns:
        True when complete.

    """
    quad_raster_path = os.path.join(
        TRAINING_IMAGERY_DIR, os.path.basename(quad_uri))
    copy_from_gs(quad_uri, quad_raster_path)
    quad_info = pygeoprocessing.get_raster_info(quad_raster_path)

    # extract the bounding boxes
    bb_srs = osr.SpatialReference()
    bb_srs.ImportFromEPSG(4326)

    bounding_box_blob_list = _execute_sqlite(
        '''
        SELECT bounding_box
        FROM quad_bounding_box_uri_table
        WHERE quad_id=?
        ''', dams_database_path, argument_list=[quad_id], fetch='all')
    bounding_box_rtree = rtree.index.Index()
    for index, bounding_box_blob in enumerate(bounding_box_blob_list):
        bounding_box = pickle.loads(bounding_box_blob)

        local_bb = pygeoprocessing.transform_bounding_box(
            bounding_box, bb_srs.ExportToWkt(),
            quad_info['projection'], edge_samples=11)

        inv_gt = gdal.InvGeoTransform(quad_info['geotransform'])
        ul_i, ul_j = [int(x) for x in gdal.ApplyGeoTransform(
            inv_gt, local_bb[0], local_bb[1])]
        lr_i, lr_j = [int(x) for x in gdal.ApplyGeoTransform(
            inv_gt, local_bb[2], local_bb[3])]
        ul_i, lr_i = sorted([ul_i, lr_i])
        ul_j, lr_j = sorted([ul_j, lr_j])

        bounding_box_rtree.insert(index, [ul_i, ul_j, lr_i, lr_j])

    quad_info = pygeoprocessing.get_raster_info(quad_raster_path)
    n_cols, n_rows = quad_info['raster_size']
    for xoff in range(0, n_cols, TRAINING_IMAGE_DIMS[0]):
        xwin_size = TRAINING_IMAGE_DIMS[0]
        if xoff + xwin_size >= n_cols:
            xoff = n_cols-xwin_size-1
        for yoff in range(0, n_rows, TRAINING_IMAGE_DIMS[1]):
            ywin_size = TRAINING_IMAGE_DIMS[1]
            if yoff + ywin_size >= n_rows:
                yoff = n_rows-ywin_size-1

            local_bbs = list(bounding_box_rtree.intersection(
                xoff, yoff, xoff+xwin_size, yoff+ywin_size))
            if local_bbs:
                pass
                # TODO: clip out the png
                # TODO: transform local bbs so they're relative to the png
                # TODO: update the database with the annotations

    os.remove(quad_raster_path)


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


def main():
    """Entry point."""
    for dir_path in [
            WORKSPACE_DIR, ECOSHARD_DIR, CHURN_DIR, TRAINING_IMAGERY_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    task_graph = taskgraph.TaskGraph(
        WORKSPACE_DIR, multiprocessing.cpu_count(), 5.0)

    planet_quad_dams_database_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(PLANET_QUAD_DAMS_DATABASE_URI))
    quad_db_dl_task = task_graph.add_task(
        func=copy_from_gs,
        args=(
            PLANET_QUAD_DAMS_DATABASE_URI,
            planet_quad_dams_database_path),
        task_name='download planet quad db',
        target_path_list=[planet_quad_dams_database_path])

    task_graph.add_task(
        func=create_status_database,
        args=(planet_quad_dams_database_path, STATUS_DATABASE_PATH),
        target_path_list=[STATUS_DATABASE_PATH],
        dependent_task_list=[quad_db_dl_task],
        task_name='create status database')

    task_graph.join()

    make_training_data(
        task_graph, STATUS_DATABASE_PATH,
        TRAINING_IMAGERY_DIR, ANNOTATIONS_CSV_PATH, CLASSES_CSV_PATH)

    task_graph.close()
    task_graph.join()


def make_quad_png(quad_uri, quad_raster_path, quad_png_path):
    """Make a PNG out of a geotiff.

    Parameters:
        quad_uri (str): uri to GS bucket to dowload tif.
        quad_raster_path (str): path to target download location.
        quad_png_path (str): path to target png file.

    Returns:
        None.

    """
    copy_from_gs(quad_uri, quad_raster_path)
    raster = gdal.OpenEx(quad_raster_path, gdal.OF_RASTER)
    raster_array = raster.ReadAsArray()
    row_count, col_count = raster_array.shape[1::]
    image_2d = numpy.transpose(
        raster_array, axes=[0, 2, 1]).reshape(
        (-1,), order='F').reshape((-1, col_count*4))
    png.from_array(image_2d, 'RGBA').save(quad_png_path)
    return quad_png_path


if __name__ == '__main__':
    main()
