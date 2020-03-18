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
import shapely.geometry
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
    filename='log.txt',
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)

TRAINING_IMAGE_DIMS = (419, 419)
MIN_BB_SIZE = 16


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
            bounding_box_to_mosaic.quad_id,
            quad_id_to_uri.quad_uri,
            bounding_box
        FROM bounding_box_to_mosaic
        INNER JOIN quad_id_to_uri ON
            bounding_box_to_mosaic.quad_id = quad_id_to_uri.quad_id
        ''', quads_database_path, argument_list=[], fetch='all')

    quad_id_list = [
        [y] for y in set([x[0] for x in bounding_box_quad_uri_list])]

    # processed quads table
    # annotations
    create_database_sql = (
        """
        CREATE TABLE quad_bounding_box_uri_table (
            quad_id TEXT NOT NULL,
            quad_uri TEXT NOT NULL,
            bounding_box BLOB NOT NULL);

        CREATE INDEX quad_bounding_box_uri_table_quad_id
        ON quad_bounding_box_uri_table (quad_id);

        CREATE TABLE quad_processing_status (
            quad_id TEXT PRIMARY KEY,
            processed INT NOT NULL);

        CREATE TABLE annotation_table (
            record TEXT PRIMARY KEY);
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
        argument_list=quad_id_list, mode='modify', execute='many')


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


def make_training_data(task_graph, dams_database_path, imagery_dir):
    """Make training data by fetching imagery and building CSVs.

    Parameters:
        task_graph (taskgraph.Taskgraph): TaskGraph object to help with
            scheduling downloads.
        dams_database_path (str): path to database containing a
            "bounding_box_to_mosaic" table.
        imagery_dir (str): path to directory to store images.

    Returns:
        None

    """
    # get all the quad_ids
    quad_id_uris_to_process = _execute_sqlite(
        '''
        SELECT
            quad_bounding_box_uri_table.quad_id,
            quad_bounding_box_uri_table.quad_uri
        FROM quad_processing_status
        INNER JOIN quad_bounding_box_uri_table ON
            quad_processing_status.quad_id=quad_bounding_box_uri_table.quad_id
        WHERE processed=0
        GROUP BY quad_bounding_box_uri_table.quad_id, quad_uri
        ''', dams_database_path, argument_list=[], fetch='all')
    for index, (quad_id, quad_uri) in enumerate(quad_id_uris_to_process):
        _ = task_graph.add_task(
            func=process_quad,
            args=(quad_uri, quad_id, dams_database_path),
            transient_run=True,
            ignore_path_list=[dams_database_path],
            task_name='process quad %s' % quad_id)
        if index > multiprocessing.cpu_count() * 5:
            break
    task_graph.close()
    task_graph.join()


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
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1)
    quad_raster_path = os.path.join(
        TRAINING_IMAGERY_DIR, os.path.basename(quad_uri))
    download_quad_task = task_graph.add_task(
        func=copy_from_gs,
        args=(quad_uri, quad_raster_path),
        target_path_list=[quad_raster_path],
        task_name='download %s' % quad_uri)
    download_quad_task.join()
    quad_info = pygeoprocessing.get_raster_info(quad_raster_path)
    n_cols, n_rows = quad_info['raster_size']

    # extract the bounding boxes
    bb_srs = osr.SpatialReference()
    bb_srs.ImportFromEPSG(4326)

    bounding_box_blob_list = _execute_sqlite(
        '''
        SELECT bounding_box
        FROM quad_bounding_box_uri_table
        WHERE quad_id=?
        ''', dams_database_path, argument_list=[quad_id], fetch='all')
    working_dam_bb_list = []  # will be used to collapose duplicates later
    for index, (bounding_box_blob,) in enumerate(bounding_box_blob_list):
        bounding_box = pickle.loads(bounding_box_blob)
        LOGGER.debug('%s: %s', quad_uri, bounding_box)

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

        # possible that the sample is so small its too small. Lets make it at
        # least 16x16
        if lr_i-ul_i < MIN_BB_SIZE:
            delta = MIN_BB_SIZE - lr_i-ul_i
            ul_i -= max(1, delta//2)
            lr_i += max(1, delta//2)

        if lr_j-ul_j < MIN_BB_SIZE:
            delta = MIN_BB_SIZE - lr_j-ul_j
            ul_j -= max(1, delta//2)
            lr_j += max(1, delta//2)

        # possible the dam may lie outside of the quad, if so clip to the
        # edge of the quad
        if ul_j < 0:
            ul_j = 0
        if ul_i < 0:
            ul_i = 0
        if lr_i >= n_cols:
            lr_i = n_cols - 1
        if lr_j >= n_rows:
            lr_j = n_rows - 1

        dam_bb = [ul_i, ul_j, lr_i, lr_j]

        # this is a sanity check
        if ul_i >= n_cols or ul_j >= n_rows or lr_i < 0 or lr_j < 0:
            raise ValueError(
                'transformed coordinates outside of raster bounds: '
                'lat/lng: %s\nlocal: %sraster_bb: %s\ntransformed: %s' % (
                    bounding_box, local_bb, quad_info['bounding_box'], dam_bb))

        working_dam_bb_list.append(dam_bb)

    bounding_box_rtree = rtree.index.Index()
    index_to_bb_list = []
    while working_dam_bb_list:
        current_bb = shapely.geometry.box(*working_dam_bb_list.pop())
        for index in range(len(working_dam_bb_list)-1, -1, -1):
            test_bb = shapely.geometry.box(*working_dam_bb_list[index])
            if current_bb.intersects(test_bb):
                current_bb = current_bb.union(test_bb)
                del working_dam_bb_list[index]
        LOGGER.debug(
            'going to insert this: %s',
            str((len(index_to_bb_list), current_bb.bounds)))
        bounding_box_rtree.insert(len(index_to_bb_list), current_bb.bounds)
        index_to_bb_list.append(current_bb.bounds)

    quad_slice_index = 0
    annotation_string_list = []
    for xoff in range(0, n_cols, TRAINING_IMAGE_DIMS[0]):
        win_xsize = TRAINING_IMAGE_DIMS[0]
        if xoff + win_xsize >= n_cols:
            xoff = n_cols-win_xsize-1
        for yoff in range(0, n_rows, TRAINING_IMAGE_DIMS[1]):
            win_ysize = TRAINING_IMAGE_DIMS[1]
            if yoff + win_ysize >= n_rows:
                yoff = n_rows-win_ysize-1

            bb_indexes = list(bounding_box_rtree.intersection(
                (xoff, yoff, xoff+win_xsize, yoff+win_ysize)))

            # see if any of the bounding boxes intersect in which case make
            # a single big one

            if bb_indexes:
                LOGGER.debug(
                    'these local bbs at %d %d: %s', xoff, yoff,
                    str(bb_indexes))
                # clip out the png and name after number of bbs per image
                quad_png_path = os.path.join(
                    TRAINING_IMAGERY_DIR, '%d_%s_%d.png' % (
                        len(bb_indexes), quad_id, quad_slice_index))
                quad_slice_index += 1
                try:
                    make_quad_png(
                        quad_raster_path, quad_png_path,
                        xoff, yoff, win_xsize, win_ysize)
                    # transform local bbs so they're relative to the png
                    for bb_index in bb_indexes:
                        base_bb = list(index_to_bb_list[bb_index])
                        base_bb[0] = max(0, base_bb[0]-xoff)
                        base_bb[1] = max(0, base_bb[1]-yoff)
                        base_bb[2] = \
                            min(TRAINING_IMAGE_DIMS[0], base_bb[2]-xoff)
                        base_bb[3] = \
                            min(TRAINING_IMAGE_DIMS[1], base_bb[3]-yoff)
                        annotation_string_list.append(
                            ['%s,%d,%d,%d,%d,dam' % (
                                quad_png_path, base_bb[0], base_bb[1],
                                base_bb[2], base_bb[3])])
                except Exception:
                    LOGGER.exception('skipping %s' % quad_raster_path)

    LOGGER.debug(
        'updating annotation table with this: %s', str(annotation_string_list))
    _execute_sqlite(
        '''
        INSERT OR REPLACE INTO annotation_table
            (record)
        VALUES (?);
        ''', dams_database_path,
        argument_list=annotation_string_list, execute='many', mode='modify')

    _execute_sqlite(
        '''
        UPDATE quad_processing_status
            SET processed=1
        WHERE quad_id=?
        ''', dams_database_path, argument_list=[quad_id], mode='modify')

    task_graph.join()
    task_graph.close()
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
        hash_target_files=False,
        target_path_list=[STATUS_DATABASE_PATH],
        dependent_task_list=[quad_db_dl_task],
        task_name='create status database')

    task_graph.join()

    make_training_data(task_graph, STATUS_DATABASE_PATH, TRAINING_IMAGERY_DIR)

    annotations_list = _execute_sqlite(
        '''
        SELECT
            record
        FROM annotation_table
        ''', STATUS_DATABASE_PATH, argument_list=[], fetch='all')

    with open(CLASSES_CSV_PATH, 'w') as classes_csv_file:
        classes_csv_file.write('dam,0\n')
    annotations_string = '\n'.join([x[0] for x in annotations_list])
    LOGGER.debug('annotations list: %s', str(annotations_list))
    LOGGER.debug('annotations string: %s', annotations_string)
    with open(ANNOTATIONS_CSV_PATH, 'w') as annotations_csv_file:
        annotations_csv_file.write(annotations_string)
        annotations_csv_file.write('\n')

    task_graph.close()
    task_graph.join()


def make_quad_png(
        quad_raster_path, quad_png_path, xoff, yoff, win_xsize, win_ysize):
    """Make a PNG out of a geotiff.

    Parameters:
        quad_raster_path (str): path to target download location.
        quad_png_path (str): path to target png file.
        xoff (int): x offset to read quad array
        yoff (int): y offset to read quad array
        win_xsize (int): size of x window
        win_ysize (int): size of y window

    Returns:
        None.

    """
    raster = gdal.OpenEx(quad_raster_path, gdal.OF_RASTER)
    rgba_array = numpy.array([
        raster.GetRasterBand(i).ReadAsArray(
            xoff=xoff, yoff=yoff, win_xsize=win_xsize, win_ysize=win_ysize)
        for i in [1, 2, 3, 4]])
    try:
        row_count, col_count = rgba_array.shape[1::]
        image_2d = numpy.transpose(
            rgba_array, axes=[0, 2, 1]).reshape(
            (-1,), order='F').reshape((-1, col_count*4))
        png.from_array(image_2d, 'RGBA').save(quad_png_path)
        return quad_png_path
    except Exception:
        LOGGER.exception(
            'error on %s generate png with array:\n%s\ndims:%s\n'
            'file exists:%s\nxoff=%d, yoff=%d, win_xsize=%d, win_ysize=%d' % (
                quad_raster_path, rgba_array, rgba_array.shape,
                os.path.exists(quad_raster_path),
                xoff, yoff, win_xsize, win_ysize))
        raise


if __name__ == '__main__':
    main()
    LOGGER.debug("ALL DONE!")
