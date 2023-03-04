import numpy as np
import math
import torch
from kornia.geometry.depth import depth_to_normals
from pytorch_lightning import LightningModule

import losses.loss_functions as LossF
from models.DepthNet import DepthNet
from models.PoseNet import PoseNet
from utils import visualize_depth, visualize_image


class SCDepthModuleV3(LightningModule):
    def __init__(self, hparams):
        super(SCDepthModuleV3, self).__init__()
        self.save_hyperparameters()

        # model
        self.depth_net = DepthNet(self.hparams.hparams.resnet_layers)
        self.pose_net = PoseNet()
        self.val_epoch_losses = []
        self.train_epoch_losses = []

    def configure_optimizers(self):
        optim_params = [
            {'params': self.depth_net.parameters(), 'lr': self.hparams.hparams.lr},
            {'params': self.pose_net.parameters(), 'lr': self.hparams.hparams.lr}
        ]
        optimizer = torch.optim.Adam(optim_params)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        tgt_img, tgt_pseudo_depth, ref_imgs, intrinsics, scene_id = batch
        #print("SCENEID = ", scene_id)

        # network forward
        tgt_depth = self.depth_net(tgt_img)
        ref_depths = [self.depth_net(im) for im in ref_imgs]
        
        poses = [self.pose_net(tgt_img, im) for im in ref_imgs]
        poses_inv = [self.pose_net(im, tgt_img) for im in ref_imgs]

        # compute normal
        tgt_normal = depth_to_normals(tgt_depth, intrinsics)
        tgt_pseudo_normal = depth_to_normals(tgt_pseudo_depth, intrinsics)

        # compute loss
        w1 = self.hparams.hparams.photo_weight
        w2 = self.hparams.hparams.geometry_weight
        w3 = self.hparams.hparams.normal_matching_weight
        w4 = self.hparams.hparams.mask_rank_weight
        w5 = self.hparams.hparams.normal_rank_weight

        loss_1, loss_2, dynamic_mask = LossF.photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths,
                                                    intrinsics, poses, poses_inv, self.hparams.hparams)

        # normal_l1_loss
        loss_3 = (tgt_normal-tgt_pseudo_normal).abs().mean()

        # mask ranking loss
        loss_4 = LossF.mask_ranking_loss(tgt_depth, tgt_pseudo_depth, dynamic_mask)

        # normal ranking loss
        loss_5 = LossF.normal_ranking_loss(tgt_pseudo_depth, tgt_img, tgt_normal, tgt_pseudo_normal)
        
        loss = w1*loss_1 + w2*loss_2 + w3*loss_3 + w4*loss_4 + w5*loss_5
        
        # create logs
        self.log('train/total_loss', loss)
        self.log('train/photo_loss', loss_1)
        self.log('train/geometry_loss', loss_2)
        self.log('train/normal_l1_loss', loss_3)
        self.log('train/mask_ranking_loss', loss_4)
        self.log('train/normal_ranking_loss', loss_5)
        
        #print(f"Batch: {batch_idx}, "
        #      f"Photometric Loss (pl): {loss_1}, "
        #      f"Geometric Loss (gl): {loss_2}, "
        #      f"Normal L1 Loss (nl): {loss_3}, "
        #      f"Mask Ranking Loss (mrl): {loss_4}, "
        #      f"Normal Ranking Loss (nrl): {loss_5}, "
        #      f"Weighted Loss ({w1}*pl + {w2}*gl + {w3}*nl + {w4}*mrl + {w5}*nrl): {loss}")
        
        if math.isinf(loss) or math.isnan(loss) or torch.isnan(loss) or torch.isinf(loss):
            print("Loss is invalid! Will skip this training step and ignore its weight updates!")
            return None

        return loss
    
    def _shared_eval_step(self, batch, batch_idx, stage):
        
        if self.hparams.hparams.val_mode == 'depth':
            tgt_img, gt_depth = batch
            tgt_depth = self.depth_net(tgt_img)
            errs = LossF.compute_errors(gt_depth, tgt_depth, self.hparams.hparams.dataset_name)
            
            errs = {'abs_diff': errs[0], 'abs_rel': errs[1],
                    'a1': errs[6], 'a2': errs[7], 'a3': errs[8]}

        elif self.hparams.hparams.val_mode == 'photo':
            tgt_img, ref_imgs, intrinsics = batch
            tgt_depth = self.depth_net(tgt_img)
            ref_depths = [self.depth_net(im) for im in ref_imgs]
            poses = [self.pose_net(tgt_img, im) for im in ref_imgs]
            poses_inv = [self.pose_net(im, tgt_img) for im in ref_imgs]
            loss_1, loss_2 = LossF.photo_and_geometry_loss(tgt_img, ref_imgs, tgt_depth, ref_depths,
                                                        intrinsics, poses, poses_inv)
            errs = {'photo_loss': loss_1.item(), 'geometry_loss': loss_2.item()}
        else:
            print('wrong validation mode')
   
        if self.global_step < 10:
            return errs

        # plot 
        if batch_idx < 3:
            vis_img = visualize_image(tgt_img[0]) # (3, H, W)
            vis_depth = visualize_depth(tgt_depth[0,0]) # (3, H, W)
            stack = torch.cat([vis_img, vis_depth], dim=1).unsqueeze(0) # (3, 2*H, W)
            self.logger.experiment.add_images('{}/img_depth_{}'.format(stage, batch_idx), stack, self.current_epoch)
        
        return errs
        
    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, 'test')

    def training_epoch_end(self, outputs):
        mean_loss = torch.mean(torch.tensor([x['loss'] for x in outputs]))
        self.train_epoch_losses.append(mean_loss)

    def validation_epoch_end(self, outputs):

        if self.hparams.hparams.val_mode == 'depth':
            mean_rel = np.array([x['abs_rel'] for x in outputs]).mean()
            mean_diff = np.array([x['abs_diff'] for x in outputs]).mean()
            mean_a1 = np.array([x['a1'] for x in outputs]).mean()
            mean_a2 = np.array([x['a2'] for x in outputs]).mean()
            mean_a3 = np.array([x['a3'] for x in outputs]).mean()
            
            self.log('val_loss', mean_rel, prog_bar=True)
            self.log('val/abs_diff', mean_diff)
            self.log('val/abs_rel', mean_rel)
            self.log('val/a1', mean_a1, on_epoch=True)
            self.log('val/a2', mean_a2, on_epoch=True)
            self.log('val/a3', mean_a3, on_epoch=True)
            self.val_epoch_losses.append(mean_rel)

        elif self.hparams.hparams.val_mode == 'photo':
            mean_pl = np.array([x['photo_loss'] for x in outputs]).mean()
            self.log('val_loss', mean_pl, prog_bar=True)
  