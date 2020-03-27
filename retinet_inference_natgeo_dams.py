"""
Adapted from: https://github.com/fizyr/keras-retinanet
"""

import argparse
import collections
import logging
import multiprocessing
import os
import pathlib
import re
import subprocess
import sqlite3
import sys

from keras_retinanet import models
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.tf_version import check_tf_version
from osgeo import gdal
import cv2
import keras
import PIL
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
QUAD_CACHE_DB_PATH = 'planet_quad_cache_workspace/quad_uri.db'
WORK_DATABASE_PATH = os.path.join(CHURN_DIR, 'natgeo_dams_database.db')
logging.basicConfig(
    filename='log.txt',
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)

ISO_CODES_TO_SKIP = ['ATA']
PLANET_API_KEY_FILE = 'planet_api_key.txt'
MOSAIC_ID = '4ce5863a-fb3f-4cad-a899-b8c053af1858'


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
    # We deliberately don't use cv2.imread here, since it gives no feedback on errors while reading the image.
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
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already

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


def create_work_database(country_vector_path, target_work_database_path):
    """Create a runtime status database if it doesn't exist.

    Parameters:
        country_vector_path (str): path to a country vector with 'iso3' field.
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
            country_list TEXT NOT NULL);

        CREATE INDEX lng_min_detected_dams_index ON detected_dams (lng_min);
        CREATE INDEX lat_min_detected_dams_index ON detected_dams (lat_min);
        CREATE INDEX lng_max_detected_dams_index ON detected_dams (lng_max);
        CREATE INDEX lat_max_detected_dams_index ON detected_dams (lat_max);
        """)
    if os.path.exists(target_work_database_path):
        os.remove(target_work_database_path)
    connection = sqlite3.connect(target_work_database_path)
    connection.executescript(create_database_sql)
    connection.commit()
    connection.close()

    country_vector = gdal.OpenEx(country_vector_path, gdal.OF_VECTOR)
    country_layer = country_vector.GetLayer()

    country_index = rtree.index.Index()
    country_geom_list = []
    country_iso3_list = []
    for country_feature in country_layer:
        country_geom = country_feature.GetGeometryRef()
        country_shapely = shapely.wkb.loads(country_geom.ExportToWkb())
        country_geom = None
        country_index.insert(len(country_geom_list), country_shapely.bounds)
        country_geom_list.append(country_shapely)
        country_iso3_list.append(country_feature.GetField('iso3'))

    grid_insert_args = []
    grid_id = 0
    for lat_max in range(-60, 60):
        LOGGER.debug(lat_max)
        for lng_min in range(-180, 180):
            grid_box = shapely.geometry.box(
                lng_min, lat_max-1, lng_min+1, lat_max)
            intersecting_country_list = []
            for country_id in country_index.intersection(grid_box.bounds):
                if country_geom_list[country_id].intersects(grid_box):
                    intersecting_country_list.append(
                        country_iso3_list[country_id])
            if ISO_CODES_TO_SKIP in intersecting_country_list:
                continue
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
    dirpath = os.path.dirname(target_path)
    try:
        os.makedirs(dirpath)
    except Exception:
        pass
    subprocess.run(
        #'/usr/local/gcloud-sdk/google-cloud-sdk/bin/gsutil cp %s %s' %
        'gsutil cp %s %s' %
        (gs_uri, target_path), shell=True)


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


def main():
    """Entry point."""
    # parse arguments
    args = parse_args(sys.argv[1:])
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

    task_graph.add_task(
        func=create_work_database,
        args=(country_borders_vector_path, WORK_DATABASE_PATH,),
        hash_target_files=False,
        target_path_list=[country_borders_vector_path, WORK_DATABASE_PATH],
        dependent_task_list=[country_borders_dl_task],
        task_name='create status database')

    work_grid_list = _execute_sqlite('''
        SELECT grid_id, lng_min, lat_min, lng_max, lat_max
        FROM work_status
        WHERE country_list LIKE "%ZAF%" AND processed=0
        ''', WORK_DATABASE_PATH, argument_list=[], fetch='all')

    work_grid_list.append(
        _execute_sqlite('''
            SELECT grid_id, lng_min, lat_min, lng_max, lat_max
            FROM work_status
            WHERE country_list NOT LIKE "%ZAF%" AND processed=0
            ''', WORK_DATABASE_PATH, argument_list=[], fetch='all'))

    # find the most recent mosaic we can use
    with open(PLANET_API_KEY_FILE, 'r') as planet_api_key_file:
        planet_api_key = planet_api_key_file.read().rstrip()

    session = requests.Session()
    session.auth = (planet_api_key, '')
    for (grid_id, lng_min, lat_min, lng_max, lat_max) in work_grid_list:
        quad_id_list = get_quad_ids(
            session, MOSAIC_ID, lng_min, lat_min, lng_max, lat_max)
        for quad_id in quad_id_list:
            LOGGER.debug('attempting to read gs_uri at %s', grid_id)
            gs_uri = _execute_sqlite(
                '''
                SELECT gs_uri
                FROM quad_cache_table
                WHERE quad_id=?;
                ''', QUAD_CACHE_DB_PATH, argument_list=[grid_id], fetch='one')
            LOGGER.debug('%s: %s', quad_id, gs_uri)

    task_graph.join()
    task_graph.close()
    return

    file_to_bounding_box_list = collections.defaultdict(list)
    annotations_dir = os.path.relpath(os.path.dirname(args.annotations))
    with open(args.annotations, 'r') as annotations_file:
        for line in annotations_file:
            print(line)
            # filename_re = re.match(
            #     r'^([^,]+),(\d+),(\d+),(\d+),(\d+),', line)
            filename_re = re.match(r'^([^,]+),,,,,', line)
            if filename_re:
                file_path = os.path.join(annotations_dir, filename_re.group(1))
                file_to_bounding_box_list[file_path] = []
                print(file_path)
                # file_to_bounding_box_list[file_path].append(
                #     shapely.geometry.box(
                #         *[int(filename_re.group(i)) for i in range(2, 6)]))

    # load the model
    # make sure keras and tensorflow are the minimum required version
    check_keras_version()
    check_tf_version()

    # optionally choose specific GPU
    if args.gpu:
        setup_gpu(args.gpu)

    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone)

    # iterate through each image
    found_dams = 0
    total_detections = 0
    total_files = len(file_to_bounding_box_list)
    for file_index, (file_path, bounding_box_list) in enumerate(
            file_to_bounding_box_list.items()):
        print('file %d of %d' % (file_index+1, total_files))
        raw_image = read_image_bgr(file_path)
        image = preprocess_image(raw_image.copy())
        scale = compute_resize_scale(
            image.shape, min_side=args.image_min_side,
            max_side=args.image_max_side)
        image = cv2.resize(image, None, fx=scale, fy=scale)
        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))
        boxes, scores, labels = model.predict_on_batch(
            numpy.expand_dims(image, axis=0))[:3]
        # correct boxes for image scale
        boxes /= scale

        non_max_supression_box_list = []
        # convert box to a list from a numpy array and score to a value from
        # a single element array
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

        # no dams detected, just skip
        if not non_max_supression_box_list:
            print('nothing found')
            print(scores)
            continue

        caption_count = 0
        for box in bounding_box_list:
            draw_box(raw_image, box.bounds, (0, 0, 255), 1)
        for box, score in non_max_supression_box_list:
            total_detections += 1
            detected_box = shapely.geometry.box(*box)
            color = (255, 102, 179)
            for box in bounding_box_list:
                if box.intersects(detected_box):
                    found_dams += 1
                    color = (0, 200, 0)
                    break
            draw_box(raw_image, detected_box.bounds, color, 1)
            draw_caption(raw_image, detected_box.bounds, str(score))
            caption_count += 1

        image_path = os.path.join(
            args.save_path, '%s_annotated.png' % (
                os.path.basename(os.path.splitext(file_path)[0])))
        if caption_count > 1:
            print('check out %s' % image_path)

        cv2.imwrite(image_path, raw_image)
    print('total_detections: %d' % total_detections)
    print('found_dams: %d' % found_dams)


if __name__ == '__main__':
    main()
