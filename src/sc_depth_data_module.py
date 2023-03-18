from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler

import datasets.custom_transforms as custom_transforms
from datasets.trainset import TrainSet
from datasets.validationset import ValidationSet
from datasets.testset import TestSet


class SCDepthDataModule(LightningDataModule):

    def __init__(self, hparams,
                 selected_train_sample_indexes=None,
                 selected_val_sample_indexes=None,
                 selected_test_sample_indexes=None,
                 ):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
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
        self.test_transform = custom_transforms.Compose([
            custom_transforms.RescaleTo(self.training_size),
            custom_transforms.ArrayToTensor(),
            custom_transforms.Normalize()]
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        self.train_dataset = TrainSet(
            self.hparams.hparams.dataset_dir,
            transform=self.train_transform,
            sequence_length=self.hparams.hparams.sequence_length,
            skip_frames=self.hparams.hparams.skip_frames,
            use_frame_index=self.hparams.hparams.use_frame_index,
            with_pseudo_depth=self.load_pseudo_depth,
            selected_sample_indexes=self.selected_train_sample_indexes
        )

        if self.hparams.hparams.val_mode == 'depth':
            self.test_dataset = self.val_dataset = ValidationSet(
                self.hparams.hparams.dataset_dir,
                transform=self.valid_transform,
                dataset=self.hparams.hparams.dataset_name,
                selected_sample_indexes=self.selected_val_sample_indexes
            )
            # FIXME
            # self.test_dataset = TestSet(
            #    self.hparams.hparams.dataset_dir,
            #    transform=self.test_transform,
            #    dataset=self.hparams.hparams.dataset_name,
            #    selected_sample_indexes=self.selected_test_sample_indexes
            # )
            print("depth validation mode")
        elif self.hparams.hparams.val_mode == 'photo':
            print("photo validation mode")
            self.test_dataset = self.val_dataset = TrainSet(
                self.hparams.hparams.dataset_dir,
                transform=self.valid_transform,
                sequence_length=self.hparams.hparams.sequence_length,
                skip_frames=self.hparams.hparams.skip_frames,
                use_frame_index=self.hparams.hparams.use_frame_index,
                with_pseudo_depth=self.load_pseudo_depth,
                selected_sample_indexes=self.selected_val_sample_indexes
            )
            # self.test_dataset = TrainSet(
            #    self.hparams.hparams.dataset_dir,
            #    transform=self.test_transform,
            #    sequence_length=self.hparams.hparams.sequence_length,
            #    use_frame_index=self.hparams.hparams.use_frame_index,
            #    with_pseudo_depth=self.load_pseudo_depth,
            #    selected_sample_indexes=self.selected_test_sample_indexes
            # )
        else:
            print("wrong validation mode")

        print('{} samples found for training'.format(len(self.train_dataset)))
        print('{} samples found for validation'.format(len(self.val_dataset)))
        print('{} samples found in test'.format(len(self.test_dataset)))
        print('WARNING: test and validation datasets are currently the same. '
              'The actual test dataset will only be used in the final global model evaluation and '
              'should not be seen during the federated training.')

    def train_dataloader(self):
        print("train num_workers", self.hparams.hparams.num_workers)
        random_sampler_config = dict(data_source=self.train_dataset, replacement=True)
        if self.hparams.hparams.epoch_size > 0:
            random_sampler_config['num_samples'] = self.hparams.hparams.batch_size * self.hparams.hparams.epoch_size
        else:
            random_sampler_config['num_samples'] = len(self.train_dataset)
        print("Random Sampler Config:", random_sampler_config)
        sampler = RandomSampler(**random_sampler_config)
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

    def get_drive_ids(self, stage):
        scene_ids_by_drive_id = self.get_scene_ids_by_drive_ids(stage)
        return list(scene_ids_by_drive_id.keys())

    def get_scene_ids_by_drive_ids(self, stage):
        scene_ids = self.get_scene_ids(stage)
        scene_ids_by_drive_id = dict()
        for scene_id in scene_ids:
            drive_id = scene_id.split("_sync_")[0]
            drive_scene_ids = scene_ids_by_drive_id.get(drive_id, list())
            drive_scene_ids.append(scene_id)
            scene_ids_by_drive_id[drive_id] = drive_scene_ids
        return scene_ids_by_drive_id

    def get_samples_by_drive_id(self, stage, drive_id):
        scene_ids = self.get_scene_ids_by_drive_ids(stage)[drive_id]
        drive_samples = list()
        for scene_id in scene_ids:
            scene_samples = self.get_samples_by_scene_id(stage, scene_id)
            drive_samples.extend(scene_samples)
        return drive_samples

    def get_dataset_size(self, stage):
        if stage == 'train':
            return len(self.train_dataset)
        elif stage == 'val':
            return len(self.val_dataset)
        elif stage == 'test':
            return len(self.test_dataset)

    @staticmethod
    def get_training_size(dataset_name):
        training_size = None
        if dataset_name == 'kitti':
            training_size = [256, 832]
        elif dataset_name == 'ddad':
            training_size = [384, 640]
        elif dataset_name in ['nyu', 'tum', 'bonn']:
            training_size = [256, 320]
        else:
            print('unknown dataset type')
        return training_size
