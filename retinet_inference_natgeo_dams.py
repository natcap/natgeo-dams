"""
Adapted from: https://github.com/fizyr/keras-retinanet
"""

import argparse
import collections
import logging
import os
import multiprocessing
import pathlib
import subprocess
import sqlite3
import sys
import threading

from keras_retinanet import models
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.tf_version import check_tf_version
from osgeo import gdal
from osgeo import osr
import cv2
import ecoshard
import keras
import PIL
import png
import pygeoprocessing
import numpy
import requests
import retrying
import rtree
import shapely.geometry
import shapely.wkb
import taskgraph


WORKSPACE_DIR = 'natgeo_inference_workspace'
DETECTED_DAM_IMAGERY_DIR = os.path.join(WORKSPACE_DIR, 'detected_dam_imagery')
ECOSHARD_DIR = os.path.join(WORKSPACE_DIR, 'ecoshard')
CHURN_DIR = os.path.join(WORKSPACE_DIR, 'churn')
COUNTRY_BORDER_VECTOR_URI = (
    'gs://natgeo-dams-data/ecoshards/'
    'countries_iso3_md5_6fb2431e911401992e6e56ddf0a9bcda.gpkg')
PLANET_GRID_ID_TO_QUAD_URI = (
    'gs://natgeo-dams-data/databases/'
    'planet_cell_to_grid_md5_eb607fdb74a6278e9597fddeb59b58c1.db')
PLANET_API_KEY_FILE = 'planet_api_key.txt'
QUAD_CACHE_DB_PATH = os.path.join(
    'planet_quad_cache_workspace', 'quad_uri.db')
WORK_DATABASE_PATH = os.path.join(CHURN_DIR, 'natgeo_dams_database.db')
logging.basicConfig(
    filename='log.txt',
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)

COUNTRY_PRIORITIES = [
    'ZAF',
    'MMR',
    'COL',
    'BRA',
    'CHN',
    'CRI',
    'ZMB',
    'GHA',
    'PER']

REQUEST_TIMEOUT = 1.5
TRAINING_IMAGE_DIMS = (419, 419)

_WGS84_SRS = osr.SpatialReference()
_WGS84_SRS.ImportFromEPSG(4326)
WGS84_WKT = _WGS84_SRS.ExportToWkt()


def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    """ Compute an image scale size to constrained to min_side/max_side.
    Args
        min_side: The image's min side will be equal to min_side after
            resizing.
        max_side: If after resizing the image's max side is above max_side,
            resize until the max side is equal to max_side.
    Returns
        A resizing scale.
    """
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def read_image_bgr(path):
    """ Read an image in BGR format.
    Args
        path: Path to the image.
    """
    image = numpy.asarray(PIL.Image.open(path).convert('RGB'))
    return image[:, :, ::-1].copy()


def preprocess_image(x, mode='caffe'):
    """ Preprocess an image by subtracting the ImageNet mean.
    Args
        x: numpy.array of shape (None, None, 3) or (3, None, None).
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.
    Returns
        The input with the ImageNet mean subtracted.
    """
    # covert always to float32 to keep compatibility with opencv
    x = x.astype(numpy.float32)
    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x -= [103.939, 116.779, 123.68]
    return x


def draw_box(image, box, color, thickness):
    """ Draws a box on an image with a given color.
    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = numpy.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness)


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.
    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = numpy.array(box).astype(int)
    cv2.putText(
        image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1,
        (0, 0, 0), 2)
    cv2.putText(
        image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1,
        (255, 255, 255), 1)


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(
        description='Evaluation script for a RetinaNet network.')
    parser.add_argument('model', help='Path to RetinaNet model.')
    parser.add_argument(
        '--gpu', help='Id of the GPU to use (as reported by nvidia-smi).',
        type=int)
    return parser.parse_args(args)


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


def create_work_database(target_work_database_path, country_vector_path):
    """Create a runtime status database if it doesn't exist.

    Parameters:
        target_work_database_path (str): path to database to create.

    Returns:
        None.

    """
    LOGGER.debug('launching create_work_database')

    # processed quads table
    # annotations
    create_database_sql = (
        """
        CREATE TABLE work_status (
            grid_id INTEGER NOT NULL PRIMARY KEY,
            lng_min REAL NOT NULL,
            lat_min REAL NOT NULL,
            lng_max REAL NOT NULL,
            lat_max REAL NOT NULL,
            country_list TEXT NOT NULL,
            processed INT NOT NULL);

        CREATE INDEX lng_min_work_status_index ON work_status (lng_min);
        CREATE INDEX lat_min_work_status_index ON work_status (lat_min);
        CREATE INDEX lng_max_work_status_index ON work_status (lng_max);
        CREATE INDEX lat_max_work_status_index ON work_status (lat_max);

        CREATE TABLE detected_dams (
            dam_id INTEGER NOT NULL PRIMARY KEY,
            lng_min REAL NOT NULL,
            lat_min REAL NOT NULL,
            lng_max REAL NOT NULL,
            lat_max REAL NOT NULL,
            probability REAL NOT NULL,
            country_list TEXT NOT NULL,
            image_uri TEXT NOT NULL);

        CREATE INDEX lng_min_detected_dams_index ON detected_dams (lng_min);
        CREATE INDEX lat_min_detected_dams_index ON detected_dams (lat_min);
        CREATE INDEX lng_max_detected_dams_index ON detected_dams (lng_max);
        CREATE INDEX lat_max_detected_dams_index ON detected_dams (lat_max);
        CREATE INDEX image_uri_detected_dams_index
            ON detected_dams (image_uri);
        """)
    if os.path.exists(target_work_database_path):
        os.remove(target_work_database_path)
    connection = sqlite3.connect(target_work_database_path)
    connection.executescript(create_database_sql)
    connection.commit()
    connection.close()

    grid_insert_args = []
    grid_id = 0
    for lat_max in range(-60, 60):
        LOGGER.debug(lat_max)
        for lng_min in range(-180, 180):
            grid_box = shapely.geometry.box(
                lng_min, lat_max-1, lng_min+1, lat_max)
            intersecting_country_list = \
                get_country_intersection_list(
                    grid_box, country_vector_path)
            if intersecting_country_list:
                grid_insert_args.append((
                    grid_id, lng_min, lat_max-1, lng_min+1, lat_max,
                    ','.join(intersecting_country_list), 0))
                grid_id += 1

    _execute_sqlite(
        """
        INSERT INTO
        work_status
            (grid_id, lng_min, lat_min, lng_max, lat_max, country_list,
             processed)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, target_work_database_path,
        argument_list=grid_insert_args, mode='modify', execute='many')


def copy_from_gs(gs_uri, target_path):
    """Copy a GS objec to `target_path."""
    try:
        dirpath = os.path.dirname(target_path)
        try:
            os.makedirs(dirpath)
        except Exception:
            pass
        subprocess.run(
            'gsutil cp %s %s' %
            (gs_uri, target_path), shell=True, check=True)
    except Exception:
        LOGGER.exception('exception on copy_from_gs')
        raise


@retrying.retry(
    wait_exponential_multiplier=100, wait_exponential_max=2000,
    stop_max_attempt_number=10)
def fetch_quad(session, quad_id, quad_target_path):
    """Attempt to copy, then fetch, then upload quad."""
    try:
        quad_uri = 'gs://natgeo-dams-data/known-dam-quads/%s.tif' % quad_id
        try:
            copy_from_gs(quad_uri, quad_target_path)
        except Exception:
            # Try to download it
            get_quad_url = (
                f'https://api.planet.com/basemaps/v1/mosaics/'
                f'4ce5863a-fb3f-4cad-a899-b8c053af1858/quads/{quad_id}')
            quads_json = session.get(get_quad_url, timeout=REQUEST_TIMEOUT)
            download_url = (quads_json.json())['_links']['download']
            ecoshard.download_url(download_url, quad_target_path)
            try:
                # Try to upload it
                subprocess.run(
                    'gsutil cp %s %s'
                    % (quad_target_path, quad_uri), shell=True, check=True)
            except subprocess.CalledProcessError:
                LOGGER.warning('couldn\'t copy to bucket')
    except Exception:
        LOGGER.exception('error on quad %s' % quad_id)
        raise


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


def grid_done_worker(work_database_path, grid_done_queue):
    """Monitor if a grid is done and update if so."""
    grid_status = collections.defaultdict(int)
    try:
        while True:
            payload = grid_done_queue.get()
            if payload == 'STOP':
                break
            grid_id, count = payload
            grid_status[grid_id] += count
            LOGGER.debug('got %d work for %s', count, grid_id)
            LOGGER.debug('grid_status: %s', grid_status)
            if grid_status[grid_id] == 0:
                LOGGER.debug('all done updating database! %s', grid_id)
                del grid_status[grid_id]
                _execute_sqlite(
                    '''
                    UPDATE work_status
                    SET processed=1
                    WHERE grid_id=?;
                    ''', work_database_path,
                    mode='modify', execute='execute',
                    argument_list=[grid_id])
    except Exception:
        LOGGER.exception('error occured')
        raise


def process_quad_worker(planet_api_key, quad_queue, work_queue, grid_done_queue):
    try:
        session = requests.Session()
        session.auth = (planet_api_key, '')
        while True:
            payload = quad_queue.get()
            if payload == 'STOP':
                quad_queue.put('STOP')
                break

            grid_id, quad_id = payload
            LOGGER.debug('attempting to process quad %s', quad_id)

            # copy quad locally
            target_quad_path = os.path.join(CHURN_DIR, '%s.tif' % quad_id)
            try:
                fetch_quad(session, quad_id, target_quad_path)
            except Exception:
                LOGGER.exception("quad didn't fetch, skip it")
                continue

            quad_info = pygeoprocessing.get_raster_info(target_quad_path)
            n_cols, n_rows = quad_info['raster_size']
            # extract the bounding boxes
            quad_slice_index = 0
            for xoff in range(0, n_cols, TRAINING_IMAGE_DIMS[0]):
                win_xsize = TRAINING_IMAGE_DIMS[0]
                if xoff + win_xsize >= n_cols:
                    xoff = n_cols-win_xsize-1
                for yoff in range(0, n_rows, TRAINING_IMAGE_DIMS[1]):
                    win_ysize = TRAINING_IMAGE_DIMS[1]
                    if yoff + win_ysize >= n_rows:
                        yoff = n_rows-win_ysize-1
                    try:
                        quad_png_path = os.path.join(
                            CHURN_DIR, '%s_%d.png' % (
                                quad_id, quad_slice_index))
                        quad_slice_index += 1
                        make_quad_png(
                            target_quad_path, quad_png_path,
                            xoff, yoff, win_xsize, win_ysize)
                        LOGGER.debug(
                            'sending this to work queue: %s',
                            quad_png_path)
                        work_queue.put(
                            (grid_id, quad_png_path, xoff, yoff,
                             quad_info.copy()))
                        grid_done_queue.put((grid_id, 1))
                    except Exception:
                        LOGGER.exception(
                            'something bad happened, skipping %s'
                            % quad_png_path)

            if os.path.exists(target_quad_path):
                os.remove(target_quad_path)

    except Exception:
        LOGGER.exception('error occured')
        raise


def detect_dams_worker(work_queue, inference_queue):
    """Process work queue for image_path then pass to inference."""
    try:
        while True:
            payload = work_queue.get()
            if payload == 'STOP':
                payload.put('STOP')
                break
            grid_id, image_path, xoff, yoff, quad_info = payload

            raw_image = read_image_bgr(image_path)
            image = preprocess_image(raw_image)
            scale = compute_resize_scale(
                image.shape, min_side=800, max_side=1333)
            image = cv2.resize(image, None, fx=scale, fy=scale)
            if keras.backend.image_data_format() == 'channels_first':
                image = image.transpose((2, 0, 1))
            inference_queue.put(
                (grid_id, image, scale, image_path, xoff, yoff, quad_info))
    except Exception:
        LOGGER.exception('error occured')
        raise


def inference_worker(model, inference_queue, postprocessing_queue):
    """Do inference."""
    try:
        while True:
            payload = inference_queue.get()
            if payload == 'STOP':
                inference_queue.put('STOP')
                break
            grid_id, image, scale, image_path, xoff, yoff, quad_info = payload
            result = model.predict_on_batch(
                numpy.expand_dims(image, axis=0))[:3]
            # correct boxes for image scale
            boxes, scores, labels = result
            boxes /= scale
            postprocessing_queue.put(
                (grid_id, boxes, scores, image_path, xoff, yoff, quad_info))
    except Exception:
        LOGGER.exception('error occured')
        raise


def postprocessing_worker(
        postprocessing_queue, country_borders_vector_path, work_database_path,
        grid_done_queue):
    """Get detected images, annotate them, and stick them in the db."""
    try:
        while True:
            payload = postprocessing_queue.get()
            if payload == 'STOP':
                postprocessing_queue.put('STOP')
                break
            grid_id, boxes, scores, image_path, xoff, yoff, quad_info = payload
            non_max_supression_box_list = []
            # convert box to a list from a numpy array and score to a value
            # from a single element array
            box_score_tuple_list = [
                (list(box), score) for box, score in zip(boxes[0], scores[0])
                if score > 0.3]
            while box_score_tuple_list:
                box, score = box_score_tuple_list.pop()
                shapely_box = shapely.geometry.box(*box)
                keep = True
                # this list makes a copy
                for test_box, test_score in list(box_score_tuple_list):
                    shapely_test_box = shapely.geometry.box(*test_box)
                    if shapely_test_box.intersects(shapely_box):
                        if test_score > score:
                            # keep the new one
                            keep = False
                            break
                if keep:
                    non_max_supression_box_list.append((box, score))

            if not non_max_supression_box_list:
                # no dams detected
                os.remove(image_path)
                grid_done_queue.put((grid_id, -1))
                continue

            # if non_max_supression_box_list:
            #     LOGGER.debug('found %d dams', len(non_max_supression_box_list))
            #     raw_image = read_image_bgr(image_path)
            #     for box, score in non_max_supression_box_list:
            #         detected_box = shapely.geometry.box(*box)
            #         color = (255, 102, 179)
            #         draw_box(raw_image, detected_box.bounds, color, 1)
            #         draw_caption(raw_image, detected_box.bounds, str(score))

            #     cv2.imwrite(image_path, raw_image)
            # else:
            #     # no dams detected
            #     os.remove(image_path)
            #     grid_done_queue.put((grid_id, -1))
            #     continue

            # transform local bbs so they're relative to the png
            lng_lat_score_list = []
            for bounding_box, score in non_max_supression_box_list:
                global_bounding_box = [
                    bounding_box[0]+xoff,
                    bounding_box[1]+yoff,
                    bounding_box[2]+xoff,
                    bounding_box[3]+yoff]

                # convert to lat/lng
                geotransform = quad_info['geotransform']
                x_a, y_a = [x for x in gdal.ApplyGeoTransform(
                    geotransform, global_bounding_box[0],
                    global_bounding_box[1])]
                x_b, y_b = [x for x in gdal.ApplyGeoTransform(
                    geotransform, global_bounding_box[2],
                    global_bounding_box[3])]
                x_min, x_max = sorted([x_a, x_b])
                y_min, y_max = sorted([y_a, y_b])
                x_y_bounding_box = [
                    x_min, y_min, x_max, y_max]
                LOGGER.debug('original bounding box: %s', bounding_box)
                LOGGER.debug('xoff: %s yoff: %s', xoff, yoff)
                LOGGER.debug('global_bounding_box: %s', global_bounding_box)
                LOGGER.debug('xy bounding box: %s', x_y_bounding_box)

                lng_lat_bounding_box = \
                    pygeoprocessing.transform_bounding_box(
                        x_y_bounding_box, quad_info['projection'],
                        WGS84_WKT)
                LOGGER.debug('lng_lat_bounding_box: %s', lng_lat_bounding_box)

                # get country intersection list
                shapely_box = shapely.geometry.box(
                    *lng_lat_bounding_box)

                country_intersection_list = \
                    get_country_intersection_list(
                        shapely_box,
                        country_borders_vector_path)

                lng_lat_score_list.append((
                    lng_lat_bounding_box + [
                        float(score),
                        ','.join(country_intersection_list),
                        image_path]))

            # upload .pngs to bucket this is old code but i want to keep it
            # try:
            #     quad_uri = (
            #         'gs://natgeo-dams-data/detected_dam_data/'
            #         'annotated_imagery/%s' % os.path.basename(
            #             image_path))
            #     subprocess.run(
            #         'gsutil mv %s %s'
            #         % (image_path, quad_uri), shell=True,
            #         check=True)
            # except subprocess.CalledProcessError:
            #     LOGGER.warning(
            #         'file might already exist -- not uploading')
            #     if os.path.exists(image_path):
            #         os.remove(image_path)

            _execute_sqlite(
                """
                INSERT INTO
                detected_dams
                    (lng_min, lat_min, lng_max, lat_max, probability,
                     country_list, image_uri)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, work_database_path,
                argument_list=lng_lat_score_list, mode='modify',
                execute='many')
            grid_done_queue.put((grid_id, -1))
            try:
                os.remove(image_path)
            except Exception:
                LOGGER.exception(
                    "couldn't remove %s after postprocessing", image_path)
    except Exception:
        LOGGER.exception('error occured')
        raise


def main():
    """Entry point."""
    args = parse_args(sys.argv[1:])
    check_keras_version()
    check_tf_version()

    # optionally choose specific GPU
    if args.gpu:
        setup_gpu(args.gpu)

    model = models.load_model(args.model, backbone_name='resnet50')

    for dir_path in [
            WORKSPACE_DIR, ECOSHARD_DIR, CHURN_DIR, DETECTED_DAM_IMAGERY_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass

    task_graph = taskgraph.TaskGraph(
        WORKSPACE_DIR, -1, 5.0)

    country_borders_vector_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(COUNTRY_BORDER_VECTOR_URI))
    country_borders_dl_task = task_graph.add_task(
        func=copy_from_gs,
        args=(
            COUNTRY_BORDER_VECTOR_URI,
            country_borders_vector_path),
        task_name='download country borders vector',
        target_path_list=[country_borders_vector_path])

    planet_grid_id_to_quad_path = os.path.join(
        ECOSHARD_DIR, os.path.basename(PLANET_GRID_ID_TO_QUAD_URI))
    country_borders_dl_task = task_graph.add_task(
        func=copy_from_gs,
        args=(
            PLANET_GRID_ID_TO_QUAD_URI,
            planet_grid_id_to_quad_path),
        task_name='download planet grid to quad id db',
        target_path_list=[planet_grid_id_to_quad_path])

    if not os.path.exists(WORK_DATABASE_PATH):
        task_graph.add_task(
            func=create_work_database,
            args=(WORK_DATABASE_PATH, country_borders_vector_path),
            hash_target_files=False,
            target_path_list=[country_borders_vector_path, WORK_DATABASE_PATH],
            dependent_task_list=[country_borders_dl_task],
            task_name='create status database')

    # find the most recent mosaic we can use
    with open(PLANET_API_KEY_FILE, 'r') as planet_api_key_file:
        planet_api_key = planet_api_key_file.read().rstrip()

    quad_queue = multiprocessing.Queue(10)
    grid_done_queue = multiprocessing.Queue()
    work_queue = multiprocessing.Queue(10)
    inference_queue = multiprocessing.Queue(10)
    postprocessing_queue = multiprocessing.Queue(10)

    grid_done_worker_thread = threading.Thread(
        target=grid_done_worker,
        args=(WORK_DATABASE_PATH, grid_done_queue))
    grid_done_worker_thread.start()

    process_quad_worker_list = []
    for _ in range(1):
        process_quad_worker_process = threading.Thread(
            target=process_quad_worker,
            args=(planet_api_key, quad_queue, work_queue, grid_done_queue))
        process_quad_worker_process.start()
        process_quad_worker_list.append(process_quad_worker_process)

    detect_dams_worker_list = []
    for _ in range(1):
        detect_dams_worker_process = threading.Thread(
            target=detect_dams_worker,
            args=(work_queue, inference_queue))
        detect_dams_worker_process.start()
        detect_dams_worker_list.append(detect_dams_worker_process)

    inference_worker_thread = threading.Thread(
        target=inference_worker,
        args=(model, inference_queue, postprocessing_queue))
    inference_worker_thread.start()

    postprocessing_worker_list = []
    for _ in range(1):
        postprocessing_worker_process = threading.Thread(
            target=postprocessing_worker,
            args=(
                postprocessing_queue, country_borders_vector_path,
                WORK_DATABASE_PATH, grid_done_queue))
        postprocessing_worker_process.start()
        postprocessing_worker_list.append(postprocessing_worker_process)

    # iterate through country priorities and do '' -- all, last.
    for country_iso3 in COUNTRY_PRIORITIES + ['']:
        LOGGER.debug('***** Process country %s', country_iso3)
        work_grid_list = _execute_sqlite(
            '''
            SELECT grid_id
            FROM work_status
            WHERE country_list LIKE ? AND processed=0
            ''', WORK_DATABASE_PATH, argument_list=['%%%s%%' % country_iso3],
            fetch='all')
        for (grid_id,) in work_grid_list:
            quad_id_list = _execute_sqlite(
                '''
                SELECT quad_id_list
                FROM grid_id_to_quad_id
                WHERE grid_id=?
                ''', planet_grid_id_to_quad_path, argument_list=[grid_id],
                fetch='one')[0].split(',')
            # make the score really high
            grid_done_queue.put((grid_id, 100000))
            for quad_id in quad_id_list:
                quad_queue.put((grid_id, quad_id))
            grid_done_queue.put((grid_id, -100000))

    LOGGER.debug('waiting for quad workers to stop')
    quad_queue.put('STOP')
    for quad_worker_process in process_quad_worker_list:
        quad_worker_process.join()

    LOGGER.debug('waiting for detect dams to stop')
    work_queue.put('STOP')
    for detect_dams_process in detect_dams_worker_list:
        detect_dams_process.join()

    LOGGER.debug('waiting for inference worker to stop')
    inference_queue.put('STOP')
    inference_worker_thread.join()

    LOGGER.debug('waiting for postprocessing worker to stop')
    postprocessing_queue.put('STOP')
    for postprocessing_worker_process in postprocessing_worker_list:
        postprocessing_worker_process.join()

    grid_done_queue.put('STOP')
    grid_done_worker_thread.join()

    task_graph.join()
    task_graph.close()
    return


if __name__ == '__main__':
    main()
