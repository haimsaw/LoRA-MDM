import torch

from model.mdm import MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import get_cond_mode
import os

def load_model_wo_clip(model, state_dict):
    # assert (state_dict['sequence_pos_encoder.pe'][:model.sequence_pos_encoder.pe.shape[0]] == model.sequence_pos_encoder.pe).all()  # TEST
    # assert (state_dict['embed_timestep.sequence_pos_encoder.pe'][:model.embed_timestep.sequence_pos_encoder.pe.shape[0]] == model.embed_timestep.sequence_pos_encoder.pe).all()  # TEST
    del state_dict['sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models
    del state_dict['embed_timestep.sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') or 'sequence_pos_encoder' in k  or 'q_zero' in k for k in missing_keys])



def create_model_and_diffusion(args, data, ModelClass=MDM, DiffusionClass=SpacedDiffusion):
    model = ModelClass(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args, DiffusionClass)
    return model, diffusion


def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset in ['humanml', '100style']:
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset,
            'emb_before_mask': args.emb_before_mask, 'text_encoder_type': args.text_encoder_type,
            'pos_embed_max_len': args.pos_embed_max_len, 'mask_frames': args.mask_frames,
            'lora_finetune': args.lora_finetune, 'lora_rank': args.lora_rank, 'lora_layer': args.lora_layer, 'no_lora_q': args.no_lora_q, 'lora_ff': args.lora_ff}


def create_gaussian_diffusion(args, DiffusionClass=SpacedDiffusion):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False
    
    print(f"number of diffusion-steps: {steps}")

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    if hasattr(args, 'multi_train_mode'):
        multi_train_mode = args.multi_train_mode
    else:
        multi_train_mode = None

    return DiffusionClass(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_prior_preserv=args.lambda_prior_preserv        
    )

def load_saved_model(model, model_path, use_avg: bool=False):  # use_avg_model
    state_dict = torch.load(model_path, map_location='cpu')
    # Use average model when possible
    if use_avg and 'model_avg' in state_dict.keys():
    # if use_avg_model:
        print('loading avg model')
        state_dict = state_dict['model_avg']
    else:
        if 'model' in state_dict:
            print('loading model without avg')
            state_dict = state_dict['model']
        else:
            print('checkpoint has no avg model, loading as usual.')
    load_model_wo_clip(model, state_dict)
    return model

def load_lora_to_model(model, lora_path, use_avg: bool=False):  # use_avg_model
    if '.pt' not in lora_path:
        lora_path = find_lora_path(lora_path)
        
    state_dict = torch.load(lora_path, map_location='cpu')
    # Use average model when possible

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all(['lora' not in k or 'q_zero' in k for k in missing_keys])
    
def find_lora_path(style, base_path='save/lora'):
    for dir in os.listdir(base_path):
        model_path = os.path.join(base_path, dir, 'model000004000.pt')
        if style in dir and os.path.exists(model_path):
            return model_path
    raise Exception(f'lora for style {style} not found at {base_path}')
    