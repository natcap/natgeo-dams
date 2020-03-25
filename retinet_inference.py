"""
Adapted from: https://github.com/fizyr/keras-retinanet
"""

import argparse
import os
import sys

from keras_retinanet import models
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.anchors import make_shapes_callback
from keras_retinanet.utils.eval import evaluate
from keras_retinanet.utils.eval import _get_detections
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.tf_version import check_tf_version


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
        config=args.config,
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
    generator.compute_shapes = make_shapes_callback(model)

    # print model summary
    # print(model.summary())

    # start evaluation

    all_detections, all_inferences = _get_detections(
        generator, model, score_threshold=args.score_threshold,
        max_detections=args.max_detections, save_path=None)

    print(all_detections)
    print(all_inferences)


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
