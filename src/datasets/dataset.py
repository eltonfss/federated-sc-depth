import torch.utils.data as data


class DataSet(data.Dataset):

    def __init__(self):
        self.samples_by_scene_id = {}
        self.samples_by_drive_id = {}

    def build_samples_by_scene_id_map(self, samples):
        self.samples_by_scene_id = {}
        for sample in samples:
            scene_id = sample['scene_id']
            samples = self.get_samples_by_scene_id(scene_id)
            samples.append(sample)
            self.samples_by_scene_id[scene_id] = samples

    def build_samples_by_drive_id_map(self, samples):
        self.samples_by_drive_id = {}
        for sample in samples:
            drive_id = sample['scene_id']
            samples = self.get_samples_by_drive_id(drive_id)
            samples.append(sample)
            self.samples_by_drive_id[drive_id] = samples

    def get_samples_by_scene_id(self, scene_id):
        return self.samples_by_scene_id.get(scene_id, [])

    def get_samples_by_drive_id(self, drive_id):
        return self.samples_by_drive_id.get(drive_id, [])

    def get_scene_ids(self):
        return list(self.samples_by_scene_id.keys())

    def get_drive_ids(self):
        return list(self.samples_by_drive_id.keys())
