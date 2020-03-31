"""API for NatGeo-Microsoft AI4 Earth NatCap."""
import argparse
import logging
import os
import sys
import uuid

from keras_retinanet import models
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.tf_version import check_tf_version
import cv2
import flask
import keras
import PIL
import numpy
import shapely.geometry

APP = flask.Flask(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


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


@APP.route('/api/v1/process_image', methods=['POST'])
def process_image():
    """Invoked to do inference on a binary image.

    Body of the post includes a url to the stored .zip file of the archive.

    Returns
        None.

    """
    try:
        image_id = uuid.uuid4().hex
        payload = flask.request.data
        image_path = os.path.join('/', 'usr', 'local', 'images', '%s.png' % image_id)
        try:
            os.makedirs(os.path.dirname(image_path))
        except OSError:
            pass
        with open(image_path, 'wb') as png_file:
            png_file.write(payload)
        LOGGER.debug('payload written to: %s', image_path)
        LOGGER.debug('preprocess image')
        raw_image = read_image_bgr(image_path)
        image = preprocess_image(raw_image)
        scale = compute_resize_scale(
            image.shape, min_side=800, max_side=1333)
        image = cv2.resize(image, None, fx=scale, fy=scale)
        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        LOGGER.debug('run inference on image %s', str(image.shape))
        result = model.predict_on_batch(
            numpy.expand_dims(image, axis=0))
        # correct boxes for image scale
        LOGGER.debug('inference complete')
        boxes, scores, labels = result
        boxes /= scale

        # convert box to a list from a numpy array and score to a value
        # from a single element array
        non_max_supression_box_list = []
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
                non_max_supression_box_list.append((
                    [float(x) for x in box],
                    [float(x) for x in score]))

        LOGGER.debug(non_max_supression_box_list)
        return flask.jsonify(non_max_supression_box_list)
    except Exception as e:
        LOGGER.exception('error on processing image')
        return str(e), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NCI NDR Analysis.')
    parser.add_argument(
        '--app_port', type=int, default=8080,
        help='port to listen on for requests')
    args = parser.parse_args()

    LOGGER.debug('set up model')

    check_keras_version()
    check_tf_version()

    model = models.load_model(
        'natgeo_dams_model_resnet50_csv_64.h5', backbone_name='resnet50')

    LOGGER.debug('start the APP')
    APP.run(
        host='0.0.0.0',
        port=args.app_port)
