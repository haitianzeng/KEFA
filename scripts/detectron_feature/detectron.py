# The following code is largely borrowed from
# https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py and
# https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py

import argparse
import time

import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data.catalog import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode, Visualizer
import detectron2.data.transforms as T
from detectron2.structures.image_list import ImageList
import torch.nn.functional as F

import os
import glob
import pickle
from PIL import Image

from detectron_args import get_args

import warnings
warnings.filterwarnings("ignore")

coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # potted plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # dining-table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
}

class Detector():
    def __init__(self, args):
        string_args = """
                    --config-file r2r_src/detection_cfg/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
                    --input input1.jpeg
                    --confidence-threshold {}
                    --opts MODEL.WEIGHTS
                    detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
                    """.format(args.sem_pred_prob_thr)

        if args.sem_gpu_id == -2:
            string_args += """ MODEL.DEVICE cpu"""
        else:
            string_args += """ MODEL.DEVICE cuda:{}""".format(args.sem_gpu_id)

        string_args = string_args.split()

        args = get_seg_parser().parse_args(string_args)
        logger = setup_logger()
        logger.info("Arguments: " + str(args))

        self.cfg = setup_cfg(args)

        self.predictor = BatchPredictor(self.cfg)

    def get_roi_feature(self, img):
        # image_list (list of np.ndarray): a list of images of shape (H, W, C) (in BGR order).
        if isinstance(img, list):
            roi_feature = []
            for im in img:
                roi_feature.append(self.predictor([im]).detach().cpu().numpy())
            return roi_feature
        else:
            assert img.shape[2] == 3
            roi_feature = self.predictor([img]).detach().cpu().numpy()
            return [roi_feature]

    def get_roi_feature_mean_pooled(self, img):
        # image_list (list of np.ndarray): a list of images of shape (H, W, C) (in BGR order).
        if isinstance(img, list):
            roi_feature = []
            for im in img:
                feat = F.avg_pool2d(self.predictor([im]), kernel_size=(14, 14))
                roi_feature.append(feat.detach().cpu().numpy())
            return roi_feature


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = \
        args.confidence_threshold
    cfg.freeze()
    return cfg



def get_seg_parser():
    parser = argparse.ArgumentParser(
        description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class BatchPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a list of input images.

    Compared to using the model directly, this class does the following
    additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by
         `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take a list of input images

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained
            from cfg.DATASETS.TEST.

    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, image_list):
        """
        Args:
            image_list (list of np.ndarray): a list of images of
                                             shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for all images.
                See :doc:`/tutorials/models` for details about the format.
        """
        inputs = []
        for original_image in image_list:
            # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            instance = {"image": image, "height": height, "width": width}

            inputs.append(instance)

        # For a complete prediction
        # with torch.no_grad():
        #     predictions = self.model(inputs)
        #     return predictions

        # For extracting ROI features
        with torch.no_grad():
            images = self.model.preprocess_image(inputs)
            features = self.model.backbone(images.tensor)
            proposals, _ = self.model.proposal_generator(images, features)
            instances, _ = self.model.roi_heads(images, features, proposals)
            mask_features = [features[f] for f in self.model.roi_heads.in_features]
            mask_features = self.model.roi_heads.mask_pooler(mask_features, [x.pred_boxes for x in instances])

        return mask_features


if __name__ == "__main__":
    args = get_args()
    dummy_input = np.random.randn(640, 480, 3)
    # detector = Detector(args)
    # dummy_output = detector.get_roi_feature(dummy_input)

    from PIL import Image
    img_path = './data/v1/scans/gTV8FGcVJC9/matterport_skybox_images/623c2461e5c14b11832f8211a1d9ce74_skybox_small.jpg'
    img = np.asarray(Image.open(img_path))  # RGB
    detector = Detector(args)
    output = detector.get_roi_feature([img, dummy_input, img])
    print([o.shape for o in output])

    pass



