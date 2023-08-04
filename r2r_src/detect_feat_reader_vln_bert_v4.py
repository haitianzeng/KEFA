import base64
import csv
import pickle
import sys

import lmdb
import numpy as np
from tqdm import tqdm

csv.field_size_limit(2100000000)

_TSV_FIELDNAMES = [
    "scanId",
    "viewpointId",
    "image_w",
    "image_h",
    "vfov",
    "features",
    "boxes",
    "cls_prob",
    "attr_prob",
    "featureViewIndex",
    "featureHeading",
    "featureElevation",
    "viewHeading",
    "viewElevation",
]


def _convert_item(item):
    # item['scanId'] is unchanged
    # item['viewpointId'] is unchanged
    item["image_w"] = int(item["image_w"])  # pixels
    item["image_h"] = int(item["image_h"])  # pixels
    item["vfov"] = int(item["vfov"])  # degrees
    item["features"] = np.frombuffer(
        base64.b64decode(item["features"]), dtype=np.float32
    ).reshape(
        (-1, 2048)
    )  # K x 2048 region features
    item["boxes"] = np.frombuffer(
        base64.b64decode(item["boxes"]), dtype=np.float32
    ).reshape(
        (-1, 4)
    )  # K x 4 region coordinates (x1, y1, x2, y2)
    item["cls_prob"] = np.frombuffer(
        base64.b64decode(item["cls_prob"]), dtype=np.float32
    ).reshape(
        (-1, 1601)
    )  # K x 1601 region object class probabilities
    item["attr_prob"] = np.frombuffer(
        base64.b64decode(item["attr_prob"]), dtype=np.float32
    ).reshape(
        (-1, 401)
    )  # K x 401 region attribute class probabilities
    item["viewHeading"] = np.frombuffer(
        base64.b64decode(item["viewHeading"]), dtype=np.float32
    )  # 36 values (heading of each image)
    item["viewElevation"] = np.frombuffer(
        base64.b64decode(item["viewElevation"]), dtype=np.float32
    )  # 36 values (elevation of each image)
    item["featureHeading"] = np.frombuffer(
        base64.b64decode(item["featureHeading"]), dtype=np.float32
    )  # K headings for the features
    item["featureElevation"] = np.frombuffer(
        base64.b64decode(item["featureElevation"]), dtype=np.float32
    )  # K elevations for the features
    item["featureViewIndex"] = np.frombuffer(
        base64.b64decode(item["featureViewIndex"]), dtype=np.float32
    )  # K indices mapping each feature to one of the 36 images


def _get_boxes(item):
    image_width = item["image_w"]
    image_height = item["image_h"]

    boxes = item["boxes"]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area /= image_width * image_height

    N = len(boxes)
    output = np.zeros(shape=(N, 5), dtype=np.float32)

    # region encoding
    output[:, 0] = boxes[:, 0] / image_width
    output[:, 1] = boxes[:, 1] / image_height
    output[:, 2] = boxes[:, 2] / image_width
    output[:, 3] = boxes[:, 3] / image_height
    output[:, 4] = area

    return output


def _get_locations(boxes, feat_headings, feat_elevations, heading, next_heading):
    """ Convert boxes and orientation information into locations. """
    N = len(boxes)
    locations = np.ones(shape=(N, 11), dtype=np.float32)

    # region encoding
    locations[:, 0] = boxes[:, 0]
    locations[:, 1] = boxes[:, 1]
    locations[:, 2] = boxes[:, 2]
    locations[:, 3] = boxes[:, 3]
    locations[:, 4] = boxes[:, 4]

    # orientation encoding
    locations[:, 5] = np.sin(feat_headings - heading)
    locations[:, 6] = np.cos(feat_headings - heading)
    locations[:, 7] = np.sin(feat_elevations)
    locations[:, 8] = np.cos(feat_elevations)

    # next orientation encoding
    locations[:, 9] = np.sin(feat_headings - next_heading)
    locations[:, 10] = np.cos(feat_headings - next_heading)

    return locations


def load_tsv(path):
    data = []
    with open(path, "rt") as fid:
        reader = csv.DictReader(fid, delimiter="\t", fieldnames=_TSV_FIELDNAMES)
        # recast text data
        for item in tqdm(reader):
            _convert_item(item)
            data.append(item)
    return data


def load_lmdb(path):
    env = lmdb.open(path, readonly=True, readahead=False, max_readers=1, lock=False)

    with env.begin(write=False) as txn:
        keys = pickle.loads(txn.get("keys".encode()))
        data = []
        for key in tqdm(keys):
            item = pickle.loads(txn.get(key))
            _convert_item(item)
            data.append(item)
    return data


def tsv_to_lmdb(path, files):
    env = lmdb.open(path, map_size=1099511627776)
    with env.begin(write=True) as txn:
        keys = []
        for path in files:
            with open(path, "rt") as fid:
                reader = csv.DictReader(fid, delimiter="\t", fieldnames=_TSV_FIELDNAMES)
                for item in tqdm(reader):
                    key = item["scanId"] + "-" + item["viewpointId"]
                    txn.put(key.encode(), pickle.dumps(item))
                    keys.append(key.encode())
        txn.put("keys".encode(), pickle.dumps(keys))

        print(f"added {len(keys)} records to the database")


class PanoFeaturesReader_my_v4:
    def __init__(self, path, in_memory=False):
        # open database
        self.env = lmdb.open(
            path, readonly=True, readahead=False, max_readers=1, lock=False
        )

        # get keys
        with self.env.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get("keys".encode()))

        # get viewpoints
        self.viewpoints = {}
        for key in self.keys:
            scan_id, viewpoint_id = key.decode().split("-")
            if scan_id not in self.viewpoints:
                self.viewpoints[scan_id] = set()
            self.viewpoints[scan_id].add(viewpoint_id)

        # initialize memory
        self._in_memory = in_memory
        if self._in_memory:
            self.indices = set()
            # self.boxes = [None] * len(self.keys)
            self.probs = [None] * len(self.keys)
            # self.features = [None] * len(self.keys)
            self.viewids = [None] * len(self.keys)
            # self.headings = [None] * len(self.keys)
            # self.elevations = [None] * len(self.keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, key):
        # to adapt the format of key
        key = key.encode()

        # print(key)
        # print(self.keys[0])
        # print(key == self.keys[0])

        if key not in self.keys:
            raise TypeError(f"invalid key: {key}")

        index = self.keys.index(key)

        if self._in_memory and index in self.indices:
            # load from memory
            # boxes = self.boxes[index]
            probs = self.probs[index]
            # features = self.features[index]
            viewids = self.viewids[index]
            # headings = self.headings[index]
            # elevations = self.elevations[index]
        else:
            # load from disk
            with self.env.begin(write=False) as txn:
                item = pickle.loads(txn.get(key))
                _convert_item(item)

                # boxes = _get_boxes(item)
                probs = item["cls_prob"]
                # features = item["features"]
                viewids = item["featureViewIndex"]
                # headings = item["featureHeading"]
                # elevations = item["featureElevation"]

        # save to memory
        if self._in_memory and index not in self.indices:
            self.indices.add(index)
            # self.boxes[index] = boxes
            self.probs[index] = probs
            # self.features[index] = features
            self.viewids[index] = viewids
            # self.headings[index] = headings
            # self.elevations[index] = elevations

        return probs[:, 1:], viewids  # drop the first number of 1601, the rest 1600 are class probs


if __name__ == '__main__':
    features_reader = PanoFeaturesReader_my_v4(
        path="/home/haitian/Projects/data/Matterport3D/v1/features/genome/matterport-ResNet-101-faster-rcnn-genome.lmdb",
        in_memory=True,
    )

    # test_key = b'82sE5b5pLXE-b59d4c930d7a4b7f8df2f9a7ac90b424'
    test_key = '82sE5b5pLXE-b59d4c930d7a4b7f8df2f9a7ac90b424'

    probs, viewids = features_reader[test_key]

    print(probs.shape)
    print(np.max(probs[0]), np.min(probs[0]))
    print(np.max(probs), np.min(probs))

    print(viewids.shape)
    print(viewids[0])
    print(np.max(viewids), np.min(viewids))
    print(viewids)

    print('done.')

    # (100, 2048)
    # <class 'numpy.ndarray'>
    # (100, 1600)
    # 0.4188871 3.1518212e-09
    # 0.972962 3.7081277e-10
    # done.

