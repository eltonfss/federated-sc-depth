from datasets.dataset import DataSet
import numpy as np
from imageio.v2 import imread
from path import Path
import torch
from scipy import sparse


def load_sparse_depth(filename):
    sparse_depth = sparse.load_npz(filename)
    depth = sparse_depth.todense()
    return np.array(depth)


def crawl_folder(folder, dataset='nyu', selected_sample_indexes=None):

    imgs = sorted((folder/'color/').files('*.png') +
                  (folder/'color/').files('*.jpg'))

    if dataset == 'nyu':
        depths = sorted((folder/'depth/').files('*.png'))
    elif dataset == 'kitti':
        depths = sorted((folder/'depth/').files('*.npy'))
    elif dataset == 'ddad':
        depths = sorted((folder/'depth/').files('*.npz'))

    if selected_sample_indexes is not None:
        imgs = [imgs[index] for index in selected_sample_indexes]
        depths = [depths[index] for index in selected_sample_indexes]

    return imgs, depths


class TestSet(DataSet):
    """A sequence data loader where the files are arranged in this way:
        root/color/0000000.png
        root/depth/0000000.npz or png
    """

    def __init__(self, root, transform=None, dataset='nyu', selected_sample_indexes=None):
        super(DataSet, self).__init__()
        self.root = Path(root)/'testing'
        self.transform = transform
        self.dataset = dataset
        self.imgs, self.depths = crawl_folder(self.root, self.dataset, selected_sample_indexes)

    def __getitem__(self, index):
        img = imread(self.imgs[index]).astype(np.float32)

        if self.dataset == 'nyu':
            depth = torch.from_numpy(
                imread(self.depths[index]).astype(np.float32)).float()/5000
        elif self.dataset == 'kitti':
            depth = torch.from_numpy(
                np.load(self.depths[index]).astype(np.float32))
        elif self.dataset == 'ddad':
            depth = torch.from_numpy(load_sparse_depth(
                self.depths[index]).astype(np.float32))

        if self.transform is not None:
            img, _ = self.transform([img], None)
            img = img[0]

        return img, depth

    def __len__(self):
        return len(self.imgs)
