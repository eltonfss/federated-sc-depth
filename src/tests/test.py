from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets.custom_transforms as custom_transforms
from configargs import get_configargs
from sc_depth_data_module import SCDepthDataModule
from sc_depth_module_v3 import SCDepthModuleV3
from utils import *
from datasets.testset import TestSet
from losses.loss_functions import compute_errors


@torch.no_grad()
def main():
    config_args = get_configargs()

    # initialize network
    model = SCDepthModuleV3(config_args)

    # load pretrained model
    if config_args.pt_path:
        print(f"restoring trained model from {config_args.pt_path}")
        weights = torch.load(config_args.pt_path)
        model.load_state_dict(weights)
    elif config_args.ckpt_path:
        print(f"restoring trained model from {config_args.ckpt_path}")
        model = model.load_from_checkpoint(config_args.ckpt_path, strict=False)
    else:
        print(f"Testing with untrained model")

    model = model.depth_net
    model.cuda()
    model.eval()

    # get training resolution
    training_size = SCDepthDataModule.get_training_size(config_args.dataset_name)

    # data loader
    test_transform = custom_transforms.Compose([
        custom_transforms.RescaleTo(training_size),
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize()]
    )
    test_dataset = TestSet(
        config_args.dataset_dir,
        transform=test_transform,
        dataset=config_args.dataset_name
    )
    print('{} samples found in test scenes'.format(len(test_dataset)))

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=8,
                             pin_memory=True
                             )

    all_errs = []
    for i, (tgt_img, gt_depth) in enumerate(tqdm(test_loader)):
        pred_depth = model(tgt_img.cuda())

        errs = compute_errors(gt_depth.cuda(), pred_depth,
                              config_args.dataset_name)

        all_errs.append(np.array(errs))

    all_errs = np.stack(all_errs)
    mean_errs = np.mean(all_errs, axis=0)

    print("\n  " + ("{:>8} | " * 9).format("abs_diff", "abs_rel",
          "sq_rel", "log10", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 9).format(*mean_errs.tolist()) + "\\\\")


if __name__ == '__main__':
    main()
