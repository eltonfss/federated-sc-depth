from typing import List

from er_buffer import ExperienceReplayBuffer, ExperienceReplayBatchInfo
from sc_depth_module_v3 import SCDepthModuleV3
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch


class SCDepthModuleV3WithExperienceReplay(SCDepthModuleV3):

    def __init__(self, hparams, er_buffer: ExperienceReplayBuffer):
        super(SCDepthModuleV3WithExperienceReplay, self).__init__(hparams)
        self._er_buffer = er_buffer
        self._er_size = hparams.er_size
        self._er_frequency = hparams.er_frequency
        self._dataset_name = self.hparams.hparams.dataset_name
        self._batch_count = 0

    def set_er_buffer(self, er_buffer):
        self._er_buffer = er_buffer

    def training_step(self, batch, batch_idx):
        # print(f'training_step called for {batch_idx=} batch={len(batch)}')

        # train with current batch and add it to buffer
        er_losses = self._compute_er_losses(batch)
        current_loss = super(SCDepthModuleV3WithExperienceReplay, self).training_step(batch, batch_idx)
        self._batch_count += 1

        # compute mean loss
        if current_loss is not None and isinstance(current_loss, torch.Tensor) and er_losses:
            losses = [current_loss.item()]
            losses.extend(er_losses)
            mean_loss = np.mean(losses)
            current_loss.data = torch.tensor(mean_loss, dtype=torch.float)
        return current_loss

    def _compute_er_losses(self, batch) -> List[float]:
        tgt_img, tgt_pseudo_depth, ref_imgs, intrinsics, scene_id = batch
        # replay batches from buffer
        # print(f'{self._batch_count=}, {self._er_frequency=}')
        losses = []
        if self._batch_count > 0 and self._batch_count % self._er_frequency == 0:
            secondary_device = torch.device("cpu")

            def adjust_dimensions(x: ExperienceReplayBatchInfo):

                b_tgt_img, b_tgt_pseudo_depth, b_ref_imgs, b_intrinsics, b_scene_id = x.batch_data

                # adjust to current device
                b_tgt_img = b_tgt_img.to(secondary_device)
                b_tgt_pseudo_depth = b_tgt_pseudo_depth.to(secondary_device)
                b_ref_imgs = [br_ref_img.to(secondary_device) for br_ref_img in b_ref_imgs]
                b_intrinsics = b_intrinsics.to(secondary_device)

                # redimension height and width
                shape_img = tgt_img.shape[2:]
                b_tgt_img_rescaled = torch.tensor(()).new_zeros(tgt_img.shape).to(secondary_device, dtype=torch.float)
                shape_depth = tgt_pseudo_depth.shape[2:]
                b_tgt_pseudo_depth_rescaled = torch.tensor((), dtype=torch.float64).new_zeros(
                    tgt_pseudo_depth.shape).to(secondary_device, dtype=torch.float)
                shape_ref_imgs = ref_imgs[0].shape[2:]
                b_ref_imgs_rescaled = []
                b_intrinsics_rescaled = torch.tensor(()).new_zeros(intrinsics.shape).to(secondary_device, dtype=torch.float)
                for i in range(b_tgt_img.shape[0]):
                    # print(f"reshaping img from {b_tgt_img[i].shape[1:]} to {shape_img}")
                    # print(f"reshaping depth from {b_tgt_pseudo_depth[i].shape[1:]} to {shape_depth}")
                    # print(f"reshaping ref imgs from {b_ref_imgs[0][i].shape[1:]} to {shape_ref_imgs}")

                    to_pil = transforms.ToPILImage()
                    to_tensor = transforms.ToTensor()

                    img_transform = transforms.Resize(shape_img, interpolation=Image.ANTIALIAS)
                    b_tgt_img_i = torch.tensor((), dtype=torch.float64).new_zeros(tgt_img.shape[1:])
                    for k in range(len(b_tgt_img[i])):
                        b_tgt_img_i[k] = to_tensor(img_transform(to_pil(b_tgt_img[i][k])))
                    b_tgt_img_rescaled[i] = b_tgt_img_i
                    # print(f"reshaped img to {tuple(b_tgt_img_rescaled[i].shape[1:])}")
                    assert b_tgt_img_rescaled[i].shape[1:] == shape_img

                    depth_transform = transforms.Resize(shape_depth, interpolation=Image.ANTIALIAS)
                    b_tgt_pseudo_depth_i = torch.tensor((), dtype=torch.float64).new_zeros(tgt_pseudo_depth.shape[1:])
                    for k in range(len(b_tgt_pseudo_depth[i])):
                        b_tgt_pseudo_depth_i[k] = to_tensor(depth_transform(to_pil(b_tgt_pseudo_depth[i][k])))
                    b_tgt_pseudo_depth_rescaled[i] = b_tgt_pseudo_depth_i
                    # print(f"reshaped depth to {tuple(b_tgt_pseudo_depth_rescaled[i].shape[1:])}")
                    assert b_tgt_pseudo_depth_rescaled[i].shape[1:] == shape_depth

                    b_ref_imgs_i = [b_ref_img[i] for b_ref_img in b_ref_imgs]
                    ref_image_transform = transforms.Resize(shape_ref_imgs, interpolation=Image.ANTIALIAS)
                    for k in range(len(b_ref_imgs_i)):
                        if len(b_ref_imgs_rescaled) <= k:
                            b_ref_imgs_rescaled.append(torch.tensor((), dtype=torch.float64).new_zeros(
                                ref_imgs[0].shape
                            ).to(secondary_device, dtype=torch.float))
                        for j in range(len(b_ref_imgs_i[k])):
                            b_ref_imgs_rescaled[k][i][j] = to_tensor(ref_image_transform(to_pil(b_ref_imgs_i[k][j])))
                            assert b_ref_imgs_rescaled[k][i][j].shape == shape_ref_imgs
                            # print(f"reshaped ref img to {tuple(b_ref_imgs_rescaled[k][i][j].shape)}")

                    (orig_w, orig_h) = tuple(b_tgt_img[i].shape[1:])
                    (out_h, out_w) = tuple(shape_img)
                    b_intrinsics_rescaled = np.copy(b_intrinsics)
                    w_factor = (out_w * 1.0) / orig_w
                    h_factor = (out_h * 1.0) / orig_h
                    b_intrinsics_rescaled[0] *= w_factor
                    b_intrinsics_rescaled[1] *= h_factor
                    b_intrinsics_rescaled = torch.from_numpy(b_intrinsics_rescaled)
                    b_intrinsics_rescaled = b_intrinsics_rescaled.to(secondary_device, dtype=torch.float)
                    # print(f"rescaled intrinsics weight by {w_factor} and height by {h_factor}")

                x.batch_data = (
                    b_tgt_img_rescaled,
                    b_tgt_pseudo_depth_rescaled,
                    b_ref_imgs_rescaled,
                    b_intrinsics_rescaled,
                    b_scene_id
                )
                return x

            er_batches_info = self._er_buffer.get_batches_from_buffer(self._er_size, adjust_dimensions)
            primary_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            # print(f'{self._er_size=}, batches_info={len(er_batches_info)}')
            for batch_info in er_batches_info:
                # print(f"Replaying {batch_info.batch_idx=} from {batch_info.dataset_name=}")
                # print(f"Batch Length={len(batch_info.batch_data)}")
                # print(f"{batch_info.batch_data=}")
                er_tgt_img, er_tgt_pseudo_depth, er_ref_imgs, er_intrinsics, er_scene_id = batch_info.batch_data
                er_tgt_img = er_tgt_img.to(primary_device, dtype=torch.float)
                er_tgt_pseudo_depth = er_tgt_pseudo_depth.to(primary_device, dtype=torch.float)
                er_ref_imgs = [er_ref_img.to(primary_device, dtype=torch.float) for er_ref_img in er_ref_imgs]
                er_intrinsics = er_intrinsics.to(primary_device, dtype=torch.float)
                replay_batch = (er_tgt_img, er_tgt_pseudo_depth, er_ref_imgs, er_intrinsics, er_scene_id)
                loss = super(SCDepthModuleV3WithExperienceReplay, self).training_step(
                    batch=replay_batch, batch_idx=batch_info.batch_idx
                )
                if loss is not None and isinstance(loss, torch.Tensor):
                    losses.append(loss.item())
        return losses
