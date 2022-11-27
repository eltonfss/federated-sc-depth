from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

import datasets.custom_transforms as custom_transforms
from config import get_training_size
from datasets.train_folders import TrainFolder
from datasets.validation_folders import ValidationSet
from datasets.test_folder import TestSet

class VideosDataModule(LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.training_size = get_training_size(hparams.dataset_name)
        self.load_pseudo_depth = True if (
            hparams.model_version == 'v3') else False

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
            with_pseudo_depth=self.load_pseudo_depth
        )

        if self.hparams.hparams.val_mode == 'depth':
            self.val_dataset = ValidationSet(
                self.hparams.hparams.dataset_dir,
                transform=self.valid_transform,
                dataset=self.hparams.hparams.dataset_name
            )
            self.test_dataset = TestSet(
                self.hparams.hparams.dataset_dir,
                transform=self.test_transform,
                dataset=self.hparams.hparams.dataset_name
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
                with_pseudo_depth=self.load_pseudo_depth
            )
            self.test_dataset = TrainFolder(
                self.hparams.hparams.dataset_dir,
                transform=self.test_transform,
                sequence_length=self.hparams.hparams.sequence_length,
                skip_frames=self.hparams.hparams.skip_frames,
                use_frame_index=self.hparams.hparams.use_frame_index,
                with_pseudo_depth=self.load_pseudo_depth
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
    
    def get_dataset_size(self, stage):
        if stage == 'train':
            return len(self.train_dataset)
        elif stage == 'val':
            return len(self.val_dataset)
        elif stage == 'test':
            return len(self.test_dataset)
        
        