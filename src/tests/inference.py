from tqdm import tqdm
from imageio import imread, imwrite
from path import Path
from configargs import get_configargs
from sc_depth_data_module import SCDepthDataModule
from sc_depth_module_v3 import SCDepthModuleV3

import datasets.custom_transforms as custom_transforms

from utils import *


@torch.no_grad()
def main():
    config_args = get_configargs()
    sc_depth_hparams = copy.deepcopy(config_args)

    if config_args.model_version == 'v3':
        model = SCDepthModuleV3(sc_depth_hparams)
    else:
        raise Exception("model_version is invalid! Only v3 is currently supported!")

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

    # training size
    training_size = SCDepthDataModule.get_training_size(config_args.dataset_name)

    # normalization
    inference_transform = custom_transforms.Compose([
        custom_transforms.RescaleTo(training_size),
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize()]
    )

    input_dir = Path(config_args.input_dir)
    output_dir = Path(config_args.output_dir) / \
        'model_{}'.format(config_args.model_version)
    output_dir.makedirs_p()

    if config_args.save_vis:
        (output_dir/'vis').makedirs_p()

    if config_args.save_depth:
        (output_dir/'depth').makedirs_p()

    image_files = sum([(input_dir).files('*.{}'.format(ext))
                      for ext in ['jpg', 'png']], [])
    image_files = sorted(image_files)

    print('{} images for inference'.format(len(image_files)))

    for i, img_file in enumerate(tqdm(image_files)):

        filename = os.path.splitext(os.path.basename(img_file))[0]

        img = imread(img_file).astype(np.float32)
        tensor_img = inference_transform([img])[0][0].unsqueeze(0).cuda()
        pred_depth = model(tensor_img)

        if config_args.save_vis:
            vis = visualize_depth(pred_depth[0, 0]).permute(
                1, 2, 0).numpy() * 255
            imwrite(output_dir/'vis/{}.jpg'.format(filename),
                    vis.astype(np.uint8))

        if config_args.save_depth:
            depth = pred_depth[0, 0].cpu().numpy()
            np.save(output_dir/'depth/{}.npy'.format(filename), depth)


if __name__ == '__main__':
    main()
