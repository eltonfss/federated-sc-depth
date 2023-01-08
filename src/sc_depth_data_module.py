from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

import datasets.custom_transforms as custom_transforms
from datasets.train_folders import TrainFolder
from datasets.validation_folders import ValidationSet
from datasets.test_folder import TestSet

class SCDepthDataModule(LightningDataModule):

    def __init__(self, hparams, 
                 selected_train_sample_indexes=None,
                 selected_val_sample_indexes=None,
                 selected_test_sample_indexes=None,
                ):
        super().__init__()
        self.save_hyperparameters()
        self.training_size = self.get_training_size(hparams.dataset_name)
        self.load_pseudo_depth = True if (
            hparams.model_version == 'v3') else False
        self.selected_train_sample_indexes = selected_train_sample_indexes
        self.selected_val_sample_indexes = selected_val_sample_indexes
        self.selected_test_sample_indexes = selected_test_sample_indexes

        # data loader
        self.train_transform = custom_transforms.Compose([
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.RandomScaleCrop(),
            custom_transforms.RescaleTo(self.training_size),
            custom_transforms.ArrayToTensor(),
            custom_transforms.Normalize()]
        )
        self.valid_transform = custom_transforms.Compose([
            custom_transforms.RescaleTo(self.training_size),
            custom_transforms.ArrayToTensor(),
            custom_transforms.Normalize()]
        )
        self.test_transform =  custom_transforms.Compose([
            custom_transforms.RescaleTo(self.training_size),
            custom_transforms.ArrayToTensor(),
            custom_transforms.Normalize()]
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        self.train_dataset = TrainFolder(
            self.hparams.hparams.dataset_dir,
            transform=self.train_transform,
            sequence_length=self.hparams.hparams.sequence_length,
            skip_frames=self.hparams.hparams.skip_frames,
            use_frame_index=self.hparams.hparams.use_frame_index,
            with_pseudo_depth=self.load_pseudo_depth,
            selected_sample_indexes=self.selected_train_sample_indexes
        )

        if self.hparams.hparams.val_mode == 'depth':
            self.val_dataset = ValidationSet(
                self.hparams.hparams.dataset_dir,
                transform=self.valid_transform,
                dataset=self.hparams.hparams.dataset_name,
                selected_sample_indexes=self.selected_val_sample_indexes
            )
            self.test_dataset = TestSet(
                self.hparams.hparams.dataset_dir,
                transform=self.test_transform,
                dataset=self.hparams.hparams.dataset_name,
                selected_sample_indexes=self.selected_test_sample_indexes
            )
            print("depth validation mode")
        elif self.hparams.hparams.val_mode == 'photo':
            print("photo validation mode")
            self.val_dataset = TrainFolder(
                self.hparams.hparams.dataset_dir,
                transform=self.valid_transform,
                sequence_length=self.hparams.hparams.sequence_length,
                skip_frames=self.hparams.hparams.skip_frames,
                use_frame_index=self.hparams.hparams.use_frame_index,
                with_pseudo_depth=self.load_pseudo_depth,
                selected_sample_indexes=self.selected_val_sample_indexes
            )
            self.test_dataset = TrainFolder(
                self.hparams.hparams.dataset_dir,
                transform=self.test_transform,
                sequence_length=self.hparams.hparams.sequence_length,
                skip_frames=self.hparams.hparams.skip_frames,
                use_frame_index=self.hparams.hparams.use_frame_index,
                with_pseudo_depth=self.load_pseudo_depth,
                selected_sample_indexes=self.selected_test_sample_indexes
            )
        else:
            print("wrong validation mode")
            
        print('{} samples found for training'.format(len(self.train_dataset)))
        print('{} samples found for validation'.format(len(self.val_dataset)))
        print('{} samples found in test'.format(len(self.test_dataset)))

    def train_dataloader(self):
        print("train num_workers", self.hparams.hparams.num_workers)
        sampler = RandomSampler(self.train_dataset,
                                replacement=True,
                                num_samples=self.hparams.hparams.batch_size * self.hparams.hparams.epoch_size)
        return DataLoader(self.train_dataset,
                          sampler=sampler,
                          num_workers=self.hparams.hparams.num_workers,
                          batch_size=self.hparams.hparams.batch_size,
                          pin_memory=False)

    def val_dataloader(self):
        print("val num_workers", self.hparams.hparams.num_workers)
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=self.hparams.hparams.num_workers,
                          batch_size=self.hparams.hparams.batch_size,
                          pin_memory=False)

    def test_dataloader(self):
        print("test num_workers", self.hparams.hparams.num_workers)
        return DataLoader(self.test_dataset,
                          shuffle=False,
                          num_workers=self.hparams.hparams.num_workers,
                          batch_size=self.hparams.hparams.batch_size,
                          pin_memory=False)
    
    def get_scene_ids(self, stage):
        if stage == "train":
            return self.train_dataset.get_scene_ids()
        elif stage == "val":
            return self.val_dataset.get_scene_ids()
        elif stage == "test":
            return self.test_dataset.get_scene_ids()
        
    def get_samples_by_scene_id(self, stage, scene_id):
        if stage == "train":
            return self.train_dataset.get_samples_by_scene_id(scene_id)
        elif stage == "val":
            return self.val_dataset.get_samples_by_scene_id(scene_id)
        elif stage == "test":
            return self.test_dataset.get_samples_by_scene_id(scene_id)
    
    def get_dataset_size(self, stage):
        if stage == 'train':
            return len(self.train_dataset)
        elif stage == 'val':
            return len(self.val_dataset)
        elif stage == 'test':
            return len(self.test_dataset)
        
    def get_training_size(self, dataset_name):
        if dataset_name == 'kitti':
            training_size = [256, 832]
        elif dataset_name == 'ddad':
            training_size = [384, 640]
        elif dataset_name in ['nyu', 'tum', 'bonn']:
            training_size = [256, 320]
        else:
            print('unknown dataset type')
        return training_size
        
        