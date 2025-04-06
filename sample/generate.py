# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from data_loaders.humanml_utils import get_inpainting_mask
from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
from diffusion.respace import SpacedDiffusion
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_saved_model, load_lora_to_model
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion, plot_3d_motion_with_trajectories
import shutil
from data_loaders.tensors import collate
from moviepy.editor import clips_array

def main(args=None):
    if args is None:
        # args is None unless this method is called from another function (e.g. during training)
        args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml', '100style'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length*fps)) if args.motion_length is not None else max_frames
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)
    if out_path == '':
        path = args.lora_path if args.lora_path is not None else args.model_path
        out_path = os.path.join(os.path.dirname(path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 1
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        if hasattr(args, 'tokens') and '{token}' in texts[0]:
            # for style mixing
            texts = [prompt.replace('{token}', token) for token in args.tokens for prompt in texts]
        elif '{token}' in texts[0]:
            texts = [prompt.replace('{token}', 'sks') for prompt in texts]
        
        assert not('sks' not in texts[0] and args.prompt_suffix is not None)
        if args.prompt_suffix is not None:
            texts = [prompt.replace('in sks style', args.prompt_suffix) for prompt in texts]

        args.num_samples = len(texts)
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [s.replace('\n', '') for s in action_text]
        args.num_samples = len(action_text)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    style_data = load_dataset(args, max_frames, n_frames, styles=tuple(args.styles))

    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    DiffusionClass = InpaintingGaussianDiffusion if args.inpainting_mask != None  else SpacedDiffusion
    model, diffusion = create_model_and_diffusion(args, style_data, DiffusionClass=DiffusionClass)

    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)
            
    if args.lora_finetune and args.lora_path is not None:
        model.add_LoRA_adapters()
        load_lora_to_model(model, args.lora_path, use_avg=args.use_ema)
        
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
        
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    if is_using_data:
        iterator = iter(style_data)
        _, model_kwargs = next(iterator)
    else:
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        is_t2m = any([args.input_text, args.text_prompt])
        if is_t2m:
            # t2m
            collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        else:
            # a2m
            action = style_data.dataset.action_name_to_action(action_text)
            collate_args = [dict(arg, action=one_action, action_text=one_action_text) for
                            arg, one_action, one_action_text in zip(collate_args, action, action_text)]
        _, model_kwargs = collate(collate_args)

    all_motions = []
    all_rics = []
    all_inpaint = []
    all_lengths = []
    all_text = []
    
    if args.inpainting_mask != None:
        gt_data = get_dataset_loader(name='humanml', batch_size=args.batch_size, num_frames=n_frames)
        inpaint_data = iter(gt_data)


    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')
        
        if args.inpainting_mask != None:
            # for trajectory conditioning
            inpainted_motion = next(inpaint_data)[0][...,:n_frames].to(dist_util.dev())
            model_kwargs['y']['inpainted_motion'] = inpainted_motion
            model_kwargs['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(args.inpainting_mask, inpainted_motion.shape)).float().to(dist_util.dev())
            
        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop
        
        with torch.no_grad():
            sample = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            
        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = style_data.dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            ric_sample = sample
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
            
            if args.inpainting_mask != None:
                inp = model_kwargs['y']['inpainted_motion']
                inp = style_data.dataset.inv_transform(inp.cpu().permute(0, 2, 3, 1)).float()
                inp = recover_from_ric(inp, n_joints)
                inp = inp.view(-1, *inp.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]

        all_motions.append(sample.cpu().numpy())
        all_rics.append(ric_sample.cpu().numpy())

        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
        if args.inpainting_mask != None:
            all_inpaint.append(inp)
        print(f"created {len(all_motions) * args.batch_size} samples")

    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_rics = np.concatenate(all_rics, axis=0)
    all_rics = all_rics[:total_num_samples]  # [bs, njoints, 6, seqlen]
    
    if args.inpainting_mask != None:
        all_inpaint = np.concatenate(all_inpaint, axis=0)
        all_inpaint = all_inpaint[:total_num_samples]  # [bs, njoints, 6, seqlen]
    
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'rics':all_rics, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)
    max_vis_samples = 6
    num_vis_samples = min(args.num_samples, max_vis_samples)
    
    if args.inpainting_mask != None:
        animations = np.empty(shape=(args.num_samples, args.num_repetitions*2), dtype=object)
    else:
        animations = np.empty(shape=(args.num_samples, args.num_repetitions), dtype=object)

    for sample_i in range(args.num_samples):
        rep_files = []
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.batch_size + sample_i]
            length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)  # [:length]
            motion[length:-1] = motion[length-1]  # duplicate the last frame to end of motion, so all motions will be in equal length

            save_file = sample_file_template.format(sample_i, rep_i)
            # print(sample_print_template.format(caption, sample_i, rep_i, save_file))
            animation_save_path = os.path.join(out_path, save_file)
            if args.inpainting_mask != None:
                inp = all_inpaint[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)  # [:length]
                inp[length:-1] = inp[length-1]
                animations[sample_i, 2*rep_i] = plot_3d_motion_with_trajectories(animation_save_path, skeleton, inp, dataset=args.dataset, title='trajectory condition', fps=fps, painting_features=[args.inpainting_mask])
                animations[sample_i, 2*rep_i+1] = plot_3d_motion_with_trajectories(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps, painting_features=[args.inpainting_mask])
            else:
                animations[sample_i, rep_i] = plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps)
            rep_files.append(animation_save_path)

        save_multiple_samples(out_path, {'all': all_file_template}, animations, fps, max_frames)

        abs_path = os.path.abspath(out_path)
        print(f'[Done] Results are at [{abs_path}]')

    return out_path


def save_multiple_samples(out_path, file_templates,  animations, fps, max_frames):
    
    num_samples_in_out_file = 3
    n_samples = animations.shape[0]
    
    for sample_i in range(0,n_samples,num_samples_in_out_file):
        last_sample_i = min(sample_i+num_samples_in_out_file, n_samples)
        all_sample_save_file = file_templates['all'].format(sample_i, last_sample_i-1)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(f'saving {os.path.split(out_path)[1]}/{all_sample_save_file}')

        clips = clips_array(animations[sample_i:last_sample_i])
        clips.duration = max_frames/fps
        
        # import time
        # start = time.time()
        clips.write_videofile(all_sample_save_path, fps=fps, threads=4, logger=None)
        # print(f'duration = {time.time()-start}')
        
        for clip in clips.clips: 
            # close internal clips. Does nothing but better use in case one day it will do something
            clip.close()
        clips.close()  # important
 

def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def load_dataset(args, max_frames, n_frames, styles=None):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='text_only',
                              styles=tuple(styles))
    data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
