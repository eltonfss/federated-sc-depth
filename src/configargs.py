import configargparse


def get_configargs():
    parser = configargparse.ArgumentParser()

    # configuration
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--exp_name', type=str, help='experiment name')

    # dataset
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--dataset_name', type=str, default='kitti', choices=['kitti', 'nyu', 'ddad', 'bonn', 'tum'])
    parser.add_argument('--sequence_length', type=int, default=3, help='number of images for training')
    parser.add_argument('--skip_frames', type=int, default=1, help='jump sampling from video')
    parser.add_argument('--use_frame_index', action='store_true', help='filter out static-camera frames in video')

    # model
    parser.add_argument('--model_version', type=str, default='v1', choices=['v1', 'v2', 'v3'])
    parser.add_argument('--resnet_layers', type=int, default=18)
    parser.add_argument('--ckpt_path', type=str, default=None, help='pretrained checkpoint path to load')
    parser.add_argument('--pt_path', type=str, default=None, help='pretrained weights path to load')

    # loss for sc_v1
    parser.add_argument('--photo_weight', type=float, default=1.0, help='photometric loss weight')
    parser.add_argument('--geometry_weight', type=float, default=0.1, help='geometry loss weight')
    parser.add_argument('--smooth_weight', type=float, default=0.1, help='smoothness loss weight')

    # loss for sc_v2
    parser.add_argument('--rot_t_weight', type=float, default=1.0, help='rotation triplet loss weight')
    parser.add_argument('--rot_c_weight', type=float, default=1.0, help='rotation consistency loss weight')
    parser.add_argument('--val_mode', type=str, default='depth', choices=['photo', 'depth'], help='how to run validation')

    # loss for sc_v3
    parser.add_argument('--mask_rank_weight', type=float, default=0.1, help='ranking loss with dynamic mask sampling')
    parser.add_argument('--normal_matching_weight', type=float, default=0.1, help='weight for normal L1 loss')
    parser.add_argument('--normal_rank_weight', type=float, default=0.1, help='edge-guided sampling for normal ranking loss')

    # for ablation study
    parser.add_argument('--no_ssim', action='store_true', help='use ssim in photometric loss')
    parser.add_argument('--no_auto_mask', action='store_true', help='masking invalid static points')
    parser.add_argument('--no_dynamic_mask', action='store_true', help='masking dynamic regions')
    parser.add_argument('--no_min_optimize', action='store_true', help='optimize the minimum loss')

    # inference
    parser.add_argument('--input_dir', type=str, help='input image path')
    parser.add_argument('--output_dir', type=str, help='output depth path')
    parser.add_argument('--save-vis', action='store_true', help='save depth visualization')
    parser.add_argument('--save-depth', action='store_true', help='save depth with factor 1000')
    
    # federation
    parser.add_argument("--fed_train_num_rounds", type=int, default=500, help="number of rounds of federated training")
    parser.add_argument("--fed_train_num_participants", type=int, default=5, help="number of federated training participants: K")
    parser.add_argument("--fed_train_frac_participants_per_round", type=float, default=1.0, help="the fraction of federated training participants selected per round: C")
    parser.add_argument("--fed_train_num_local_epochs", "--num_epochs", type=int, default=-1, help="the number of local epochs executed by each participant per round: E")
    parser.add_argument("--fed_train_num_local_train_batches", "--epoch_size", type=int, default=-1, help="the number of batches used for training each local model per round: E")
    parser.add_argument("--fed_train_num_local_val_batches", type=int, default=-1, help="the number of batches used for validating each local model per round: E")
    parser.add_argument("--fed_train_num_local_sanity_val_steps", type=int, default=0, help="Proxy to https://lightning.ai/docs/pytorch/stable/common/trainer.html#num-sanity-val-steps")
    parser.add_argument("--fed_train_local_batch_size", "--batch_size", type=int, default=4, help="the local batch size used by each federated participant for training: B")
    parser.add_argument("--fed_train_local_learn_rate", "--lr", type=float, default=1e-4, help="the local learning rate used by each federated participant for training")
    parser.add_argument("--fed_train_iid", type=int, default=1, help="Use IID dataset distribution between federated participants. Default set to IID. Set to 0 for non-IID.")
    parser.add_argument("--fed_train_by_drive", type=int, default=0, help="Distributed dataset between federated participants based on the drives. Each participant only sees the data from one of the drives (only tested with KITTI)")
    parser.add_argument("--fed_train_by_drive_sort", type=str, default="sequential", help="Sort drives sequentially ('sequential'), randomly ('random') or by most samples ('eager') ")
    parser.add_argument("--fed_train_by_drive_redistribute_remaining", type=int, default=0, help="Redistribute remaining samples if number of participants is lower than number of drives")
    parser.add_argument("--fed_train_x_noniid", action="store_true", help="Use non i.i.d x distribution between federated participants")
    parser.add_argument("--fed_train_average_without_bn", action="store_true", help="do not load the weights for the batchnorm layers")
    parser.add_argument("--fed_train_participant_order", type=str, help="Indicate if participants should be ordered randomnly or sequentially during federated training", default="random")
    parser.add_argument('--fed_train_state_backup_dir', type=str, help='dirpath where the federated training state will be backup')
    parser.add_argument('--fed_train_state_restore_dir', type=str, help='dirpath from which the federated training state will be restored')

    
    # pytorch
    parser.add_argument("--num_workers", type=int, default=4, help="the number of dataloader workers")
    parser.add_argument("--gpu", default=None, help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument("--verbose", type=int, default=0, help="verbose")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--log_every_n_steps", type=int, default=1, help="Number of steps the Trainer waits until updating the log")
    
    return parser.parse_args()