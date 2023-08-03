import torch
import glob
import os
import torch.nn.functional as F
import pickle


def main():
    file_list = glob.glob('./data/v1/features/genome_grid_feat/*.pth')
    file_list = sorted(file_list)

    genome_grid_feat = dict()

    for idx, file_path in enumerate(file_list):
        with open(file_path, 'rb') as f:
            saved_tensor = torch.load(f)
            saved_tensor = F.avg_pool2d(input=saved_tensor, kernel_size=(15, 20)).squeeze(2).squeeze(2)

        genome_grid_feat[os.path.basename(file_path)[:-4]] = saved_tensor.detach().numpy()

        if idx % 100 == 0:
            print('{} / {} finished.'.format(idx, len(file_list)))

    print(file_list[0])
    print(saved_tensor.shape)

    with open('./genome_grid_feat.pkl', 'wb') as f:
        pickle.dump(genome_grid_feat, f)

    print('Saved to ./genome_grid_feat.pkl')


if __name__ == '__main__':
    main()

