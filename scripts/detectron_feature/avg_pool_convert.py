import pickle
import glob
import os

import torch
import torch.nn.functional as F
import numpy as np

root_dir = '/home/haitian/Projects/data/Matterport3D/v1/'
scans_dir = os.path.join(root_dir, 'scans')
feats_dir = os.path.join(root_dir, 'features', 'detectron_1')
# feats_scan_dir = os.path.join(root_dir, 'features', 'detectron_1_scan')
feats_all_dir = os.path.join(root_dir, 'features', 'detectron_1_all')


def main():
    scans_list = []
    for f in os.listdir(scans_dir):
        if os.path.isdir(os.path.join(scans_dir, f)):
            scans_list.append(f)
    print('Total scans: {}'.format(len(scans_list)))

    detect_feat_all = {}

    for scan_id in scans_list:
        pano_list = glob.glob(os.path.join(feats_dir, '{}_*.pkl'.format(scan_id)))
        pano_ids = [os.path.basename(item).split('_')[1][:-4] for item in pano_list]

        # scan_feat = {}

        for pano_id in pano_ids:
            with open(os.path.join(feats_dir, '{}_{}.pkl'.format(scan_id, pano_id)), 'rb') as pklfile:
                detect_feat = pickle.load(pklfile)
                detect_feat_pooled = []
                for view_detect_feat in detect_feat:
                    if view_detect_feat.shape[0] == 0:
                        view_detect_feat = view_detect_feat[:, :, 0, 0]
                    else:
                        view_detect_feat = F.avg_pool2d(torch.from_numpy(view_detect_feat), kernel_size=(14, 14))
                        view_detect_feat = view_detect_feat[:, :, 0, 0].detach().cpu().numpy()
                    detect_feat_pooled.append(view_detect_feat)

            # scan_feat[pano_id] = detect_feat_pooled

            detect_feat_pooled = np.concatenate(detect_feat_pooled, axis=0)

            detect_feat_all['{}_{}'.format(scan_id, pano_id)] = detect_feat_pooled

        # if not os.path.exists(feats_scan_dir):
        #     os.makedirs(feats_scan_dir)
        # with open(os.path.join(feats_scan_dir, '{}.pkl'.format(scan_id)), 'wb') as pklfile:
        #     pickle.dump(scan_feat, pklfile)

    if not os.path.exists(feats_all_dir):
        os.makedirs(feats_all_dir)
    with open(os.path.join(feats_all_dir, 'detect_feat_all.pkl'), 'wb') as pklfile:
        pickle.dump(detect_feat_all, pklfile)


if __name__ == "__main__":
    main()