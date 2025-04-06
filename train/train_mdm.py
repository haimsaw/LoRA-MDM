# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json

import torch
from data_loaders.humanml_utils import get_inpainting_mask
from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
from diffusion.respace import SpacedDiffusion
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import InpaintingDataLoader, get_dataset_loader, get_prior_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform, WandBPlatform  # required for the eval operation
        
def main():
    args = train_args()
    assert args.styles is not None or not args.lora_finetune # styles should be specified for training
    assert args.lora_finetune or args.lambda_prior_preserv == 0
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type) 
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')
    
    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)
    
    if args.lora_finetune:
        print("creating style data loader...")
        data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, styles=tuple(args.styles))
        print("creating prior data loader...", flush=True)
        prior_data = get_prior_dataset_loader(batch_size=args.batch_size, num_frames=args.num_frames)
    else:
        print("creating data loader...")
        data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames)
        prior_data = None
        
    if args.inpainting_mask != None:
        # for editing application
        data = InpaintingDataLoader(data, args.inpainting_mask)
    
    print("creating model and diffusion...")
    DiffusionClass = InpaintingGaussianDiffusion if args.inpainting_mask != None  else SpacedDiffusion

    model, diffusion = create_model_and_diffusion(args, data, DiffusionClass=DiffusionClass)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data, prior_data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
