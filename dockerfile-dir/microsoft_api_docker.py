"""API for NatGeo-Microsoft AI4 Earth NatCap."""
import argparse
import logging
import os
import sys
import uuid

import flask

APP = flask.Flask(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


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
        file_path = os.path.join('/', 'usr', 'local', 'images', '%s.png' % image_id)
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError:
            pass
        with open(file_path, 'wb') as png_file:
            png_file.write(payload)
        LOGGER.debug('payload written to: %s', file_path)
        return 'complete', 202
    except Exception as e:
        LOGGER.exception('error on processing image')
        return str(e), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NCI NDR Analysis.')
    parser.add_argument(
        '--app_port', type=int, default=8080,
        help='port to listen on for requests')
    args = parser.parse_args()

    LOGGER.debug('start the APP')
    APP.run(
        host='0.0.0.0',
        port=args.app_port)
