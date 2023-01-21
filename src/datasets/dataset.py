import torch.utils.data as data

class DataSet(data.Dataset):

    def __init__(self):
        self.samples_by_scene_id = {}
    
    def build_samples_by_scene_id_map(self, samples):
        #print("BUILD SAMPLES BY SCENE ID MAP")
        self.samples_by_scene_id = {}
        for sample in samples:
            scene_id = sample['scene_id']
            samples = self.get_samples_by_scene_id(scene_id)
            samples.append(sample)
            self.samples_by_scene_id[scene_id] = samples
       #print(len(samples), "mapped to ", len(list(self.samples_by_scene_id.keys())), "scenes!")
    
    def get_samples_by_scene_id(self, scene_id):
        return self.samples_by_scene_id.get(scene_id, [])
    
    def get_scene_ids(self):
        return list(self.samples_by_scene_id.keys())