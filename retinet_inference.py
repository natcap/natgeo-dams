"""
Adapted from: https://github.com/fizyr/keras-retinanet
"""

import argparse
import os
import sys

import cv2
import keras
from keras_retinanet import models
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.anchors import make_shapes_callback
from keras_retinanet.utils.eval import evaluate
from keras_retinanet.utils.eval import _get_detections
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.tf_version import check_tf_version
import numpy


def draw_box(image, box, color, thickness):
    """ Draws a box on an image with a given color.
    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = numpy.array(box).astype(int)
    print('B: %s' % str(b))
    print(color)
    print(thickness)
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


def create_generator(args, preprocess_image):
    """ Create retinet generator."""
    common_args = {
        'preprocess_image': preprocess_image,
    }
    validation_generator = CSVGenerator(
        args.annotations,
        args.classes,
        image_min_side=args.image_min_side,
        image_max_side=args.image_max_side,
        shuffle_groups=False,
        **common_args)
    return validation_generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser = argparse.ArgumentParser(
        description='Evaluation script for a RetinaNet network.')
    subparsers = parser.add_subparsers(
        help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument(
        'annotations',
        help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument(
        'classes', help='Path to a CSV file containing class label mapping.')

    parser.add_argument('model', help='Path to RetinaNet model.')
    parser.add_argument(
        '--backbone', help='The backbone of the model.', default='resnet50')
    parser.add_argument(
        '--gpu', help='Id of the GPU to use (as reported by nvidia-smi).',
        type=int)
    parser.add_argument(
        '--score-threshold',
        help='Threshold on score to filter detections (defaults to 0.05).',
        default=0.05, type=float)
    parser.add_argument(
        '--iou-threshold',
        help='IoU Threshold for positive detection (defaults to 0.5).',
        default=0.5, type=float)
    parser.add_argument(
        '--max-detections',
        help='Max Detections per image (defaults to 30).', default=30,
        type=int)
    parser.add_argument(
        '--save-path', help='Path for saving images with detections.')
    parser.add_argument(
        '--image-min-side',
        help='Rescale the image so the smallest side is min_side.', type=int,
        default=800)
    parser.add_argument(
        '--image-max-side',
        help='Rescale the image if the largest side is larger than max_side.',
        type=int, default=1333)
    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras and tensorflow are the minimum required version
    check_keras_version()
    check_tf_version()

    # optionally choose specific GPU
    if args.gpu:
        setup_gpu(args.gpu)

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # create the generator
    backbone = models.backbone(args.backbone)
    generator = create_generator(args, backbone.preprocess_image)

    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(args.model, backbone_name=args.backbone)

    for i in range(generator.size()):
        raw_image = generator.load_image(i)
        image = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        boxes, scores, labels = model.predict_on_batch(
            numpy.expand_dims(image, axis=0))[:3]

        # correct boxes for image scale
        boxes /= scale

        for box, score, label in zip(boxes, scores, labels):
            if score[0] < 0:
                break
            print(box)
            draw_box(raw_image, box, (255, 102, 179), 1)
            draw_caption(raw_image, box, str(score[0]))

            cv2.imwrite(
                os.path.join(args.save_path, '%d.png' % i), raw_image)

    # generator.compute_shapes = make_shapes_callback(model)

    # # print model summary
    # # print(model.summary())

    # # start evaluation

    # all_detections, all_inferences = _get_detections(
    #     generator, model, score_threshold=args.score_threshold,
    #     max_detections=args.max_detections, save_path=None)

    # print(all_detections)
    # print(all_inferences)


    # average_precisions, inference_time = evaluate(
    #     generator,
    #     model,
    #     iou_threshold=args.iou_threshold,
    #     score_threshold=args.score_threshold,
    #     max_detections=args.max_detections,
    #     save_path=args.save_path
    # )

    # # print evaluation
    # total_instances = []
    # precisions = []
    # for label, (average_precision, num_annotations) in \
    #         average_precisions.items():
    #     print(
    #         '{:.0f} instances of class'.format(num_annotations),
    #         generator.label_to_name(label),
    #         'with average precision: {:.4f}'.format(average_precision))
    #     total_instances.append(num_annotations)
    #     precisions.append(average_precision)

    # if sum(total_instances) == 0:
    #     print('No test instances found.')
    #     return

    # print(
    #     'Inference time for {:.0f} images: {:.4f}'.format(
    #         generator.size(), inference_time))

    # print(
    #     'mAP using the weighted average of precisions among classes: '
    #     '{:.4f}'.format(
    #         sum([a * b for a, b in zip(total_instances, precisions)]) /
    #         sum(total_instances)))
    # print(
    #     'mAP: {:.4f}'.format(
    #         sum(precisions) / sum(x > 0 for x in total_instances)))


if __name__ == '__main__':
    main()
