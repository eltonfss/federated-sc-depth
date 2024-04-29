from datasets.dataset import DataSet
import numpy as np
from imageio import imread
from path import Path
import random


def load_as_float(path):
    return imread(path).astype(np.float32)


def generate_sample_index(num_frames, skip_frames, sequence_length):
    sample_index_list = []
    k = skip_frames
    demi_length = (sequence_length-1)//2
    shifts = list(range(-demi_length * k,
                        demi_length * k + 1, k))
    shifts.pop(demi_length)

    if num_frames > sequence_length:
        for i in range(demi_length * k, num_frames-demi_length * k):
            sample_index = {'tgt_idx': i, 'ref_idx': []}
            for j in shifts:
                sample_index['ref_idx'].append(i+j)
            sample_index_list.append(sample_index)

    return sample_index_list


class TrainSet(DataSet):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self,
                 root,
                 sequence_length=3,
                 transform=None,
                 skip_frames=1,
                 dataset='kitti',
                 use_frame_index=False,
                 with_pseudo_depth=False,
                 selected_sample_indexes=None,
                 scene_list_file_name='train.txt'):
        super(DataSet, self).__init__()
        np.random.seed(0)
        random.seed(0)
        self.root = Path(root)/'training'
        scene_list_path = self.root/scene_list_file_name
        self.scenes = [self.root/folder[:-1]
                       for folder in open(scene_list_path)]
        self.transform = transform
        self.dataset = dataset
        self.k = skip_frames
        self.with_pseudo_depth = with_pseudo_depth
        self.use_frame_index = use_frame_index
        self.crawl_folders(sequence_length, selected_sample_indexes)

    def crawl_folders(self, sequence_length, selected_sample_indexes):
        # k skip frames
        sequence_set = []

        for scene in self.scenes:
            intrinsics = np.genfromtxt(
                scene/'cam.txt').astype(np.float32).reshape((3, 3))

            imgs = sorted(scene.files('*.jpg'))

            if self.use_frame_index:
                frame_index = [int(index)
                               for index in open(scene/'frame_index.txt')]
                imgs = [imgs[d] for d in frame_index]

            if self.with_pseudo_depth:
                pseudo_depths = sorted((scene/'leres_depth').files('*.png'))
                if self.use_frame_index:
                    pseudo_depths = [pseudo_depths[d] for d in frame_index]

            if len(imgs) < sequence_length:
                continue

            sample_index_list = generate_sample_index(
                len(imgs), self.k, sequence_length)
            for sample_index in sample_index_list:
                scene_id = scene.replace(self.root + "/", "")
                sample = {'scene_id': scene_id,
                          'intrinsics': intrinsics,
                          'tgt_img': imgs[sample_index['tgt_idx']]}
                if self.with_pseudo_depth:
                    sample['tgt_pseudo_depth'] = pseudo_depths[sample_index['tgt_idx']]

                sample['ref_imgs'] = []
                for j in sample_index['ref_idx']:
                    sample['ref_imgs'].append(imgs[j])
                
                sample['sourceIndex'] = len(sequence_set)
                sequence_set.append(sample)

        self.samples = sequence_set
        if selected_sample_indexes is not None:
            self.samples = [sequence_set[index] for index in selected_sample_indexes]

        if self.dataset == 'kitti':
            self.build_samples_by_scene_id_map(self.samples)
        else:
            self.build_samples_by_drive_id_map(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt_img'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]

        if self.with_pseudo_depth:
            tgt_pseudo_depth = load_as_float(sample['tgt_pseudo_depth'])

        if self.transform is not None:
            if self.with_pseudo_depth:
                imgs, intrinsics = self.transform(
                    [tgt_img, tgt_pseudo_depth] + ref_imgs, np.copy(sample['intrinsics']))
                tgt_img = imgs[0]
                tgt_pseudo_depth = imgs[1]
                ref_imgs = imgs[2:]
            else:
                imgs, intrinsics = self.transform(
                    [tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
                tgt_img = imgs[0]
                ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
            
        scene_id = sample['scene_id']

        if self.with_pseudo_depth:
            return tgt_img, tgt_pseudo_depth, ref_imgs, intrinsics, scene_id
        else:
            return tgt_img, ref_imgs, intrinsics, scene_id

    def __len__(self):
        return len(self.samples)
