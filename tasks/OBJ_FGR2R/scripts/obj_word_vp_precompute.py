import json
import copy
import pickle
import yaml
import numpy as np
import torch
from tslearn.metrics import dtw_path_from_metric
from numpy import array  # for solving the bug in eval()

from detect_feat_reader_vln_bert_v2 import PanoFeaturesReader_my_v2

MAX_DETECT = 50

'''
Precompute alignment of object words
'''

def main():
    split = 'train'  # 'test', 'val_seen', 'val_unseen'
    source = '../tasks/OBJ_FGR2R/data/OBJVG_FGR2R_{}.json'.format(split)
    target = '../tasks/OBJ_FGR2R/data/OBJVGv2_FGR2R_{}.json'.format(split)

    # vg detection feat
    feat_reader = PanoFeaturesReader_my_v2(
        path='/home/EnvDrop_my/data/v1/features/genome/matterport-ResNet-101-faster-rcnn-genome.lmdb',
        in_memory=True)

    with open(source, 'r') as f_:
        data = json.load(f_)

    new_data = copy.deepcopy(data)

    total_length = len(data)

    for idx, item in enumerate(data):
        # print('now:', idx)
        obs = []
        for vp in item['path']:
            obs.append({'scan': item['scan'],
                        'viewpoint': vp})

        detection_conf = np.zeros((len(obs), MAX_DETECT, 1600), dtype=np.float32)
        for i, ob in enumerate(obs):
            _, probs = feat_reader['{}-{}'.format(ob['scan'], ob['viewpoint'])]
            detection_conf[i] = probs[:MAX_DETECT, :]

        # detection_conf = torch.from_numpy(detection_conf).cuda()

        obj_vg_class_list = eval(item['obj_vg_class_list'])
        obj_word_vp_list = []  # contains obj_word_vp in all 3 instructions

        for obj_vg_class in obj_vg_class_list:
            obj_word_cls = np.asarray(obj_vg_class)  # sentence annotation of word vg class
            obj_word_pos = np.where(obj_word_cls > 0)[0]  # find the position of obj words in the sentence
            if obj_word_pos.shape[0] == 0:  # no valid object words
                obj_word_vp_list.append([])
                continue

            # calculate the pairwise distance (n_viewpoint * n_obj_word)
            distance = np.zeros((len(obs), obj_word_pos.shape[0]))
            for i_obj_word, word_pos in enumerate(obj_word_pos.tolist()):
                # take max over all detections at each viewpoint
                path_cls_prob = np.max(detection_conf[:, :, obj_word_cls[word_pos]], axis=1)
                distance[:, i_obj_word] = path_cls_prob.reshape(-1)

            # a larger detection probability means a shorter matching distances, so take a negative log
            distance = -np.log(distance)

            # calculate the alignment with DTW
            align_pairs, _ = dtw_path_from_metric(distance, metric="precomputed")  # TODO: check alignment

            align = np.zeros((len(obs), obj_word_pos.shape[0]))
            for i in range(len(align_pairs)):
                align[align_pairs[i][0], align_pairs[i][1]] = 1

            # calculate the class of each detection by the argmax prob
            # detect_cls = torch.argmax(detection_conf, dim=2)  # (a, MAX_DETECT)

            # for each object word, select a viewpoint with max prob (min dist) in matched viewpoint(s)
            obj_word_vp = np.zeros_like(obj_word_pos)
            for i_obj_word, word_pos in enumerate(obj_word_pos.tolist()):
                matched_vps = np.where(align[:, i_obj_word] == 1)[0]
                obj_word_vp[i_obj_word] = matched_vps[np.argmin(distance[align[:, i_obj_word] == 1, i_obj_word])]

            obj_word_vp_list.append(obj_word_vp.tolist())

        new_data[idx]['obj_word_vp_list'] = str(obj_word_vp_list)

        if idx > 0 and idx % 200 == 0:
            print('{}/{} finished.'.format(idx, total_length))

    with open(target, 'a') as file_:
        json.dump(new_data, file_, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()








