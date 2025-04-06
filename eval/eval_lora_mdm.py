# This code is based on https://github.com/neu-vi/SMooDi/blob/37d5a43b151e0b60c52fc4b37bddbb5923f14bb7/mld/models/modeltype/mld.py#L1449

import json
import re
from eval.tm2t import TM2TMetrics
from eval.style_classifier import load_classifier
from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from utils.model_util import create_model_and_diffusion, load_saved_model, load_lora_to_model, find_lora_path

from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.tensors import lengths_to_mask

from  data_loaders.humanml.networks.evaluator_wrapper import build_evaluators
import time
from tqdm import tqdm

class SmoodiEval():
    def __init__(self, args, lora_base_path):
        
        self.args = args
        self.lora_base_path = lora_base_path

        self.classifier, self.cassifier_styles, self.label_to_style  = load_classifier(args.classifier_style_group)
        self.style_to_label = {v:k for k,v in self.label_to_style.items()}

        badly_proccesd_styles = ['Zombie','WiggleHips', 'WhirlArms', 'WildArms', 'WildLegs', 'WideLegs']
        
        if args.styles is None:
            self.styles = self.cassifier_styles
        else:
            self.styles = args.styles
        assert len(set(self.cassifier_styles) & set(badly_proccesd_styles)) == 0

        # init model and metrics
        self.metrics = TM2TMetrics()
        
        self.text_enc, self.motion_enc, self.movement_enc = build_evaluators({
            'dataset_name': 't2m',
            'device': dist_util.dev(),
            'dim_word': 300,
            'dim_pos_ohot': 15,
            'dim_motion_hidden': 1024,
            'dim_text_hidden': 512,
            'dim_coemb_hidden': 512,
            'dim_pose': 263,
            'dim_movement_enc_hidden': 512,
            'dim_movement_latent': 512,
            'checkpoints_dir': '.',
            'unit_length': 4,
            })
                                                                             
        self.data_loader = get_dataset_loader(name='humanml', batch_size=args.batch_size, num_frames=None, split='test', hml_mode='gt')
               
        model, self.diffusion = create_model_and_diffusion(args, self.data_loader)
        
        print(f"Loading checkpoints from [{args.model_path}]...")
        load_saved_model(model, args.model_path, use_avg=args.use_ema)
        
        if args.lora_finetune:
            model.add_LoRA_adapters()
            self.style_set=False
            
        if self.args.guidance_param != 1:
            model = ClassifierFreeSampleModel(model)

        model.to(dist_util.dev())
        model.eval()
        self.model = model
  
    def _style_to_model(self, style):
        if not self.args.lora_finetune:
            self.style_set=False
            return

        if self.args.guidance_param != 1:
            model = self.model.model
        else:
            model = self.model

        load_lora_to_model(model, find_lora_path(style, base_path=self.lora_base_path), use_avg=self.args.use_ema)
        self.style_set=True
        
    def _procees_dataset(self):
        n_styles = len(self.styles)
        replace_every = len(self.data_loader) // n_styles
        style = None
        
        for i, batch in enumerate(tqdm(self.data_loader)):   
            if i % replace_every == 0:
                style_idx = (i // replace_every) % n_styles
                style = self.styles[style_idx]
                self._style_to_model(style)
                
            assert self.style_set or not self.args.lora_finetune
            rs_set = self._procces_batch(batch, style)
            self.metrics.update(
                rs_set["text_embeddings"],
                rs_set["gen_motion_embeddings"],
                rs_set["gt_motion_embeddings"],
                rs_set["lengths"],
                rs_set["our_predicted"],
                rs_set["label"],
                rs_set["joints_gen"],
                            )

    def _procces_batch(self, batch, gen_style):   
        # batch = [word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, tokens]
        
        gt_motions = batch[4].permute(0, 2,1).unsqueeze(2) # B J 1 F
        lengths = batch[5]
        word_embs = batch[0]
        pos_ohot = batch[1]
        texts = batch[2]
        text_lengths = batch[3]

        # sample.shape = # B J 1 F
        if self.args.lora_finetune:
            sample, inference_time = self._sample(texts, lengths, gt_motions)
        else:
            raise Exception('Not implamented')
        
        # style classification
        logits = self.classifier(sample, lengths.to(dist_util.dev()))

        # renorm 
        gt_motions_unnorm  = self.data_loader.dataset.inv_transform(gt_motions.cpu().permute(0, 2, 3, 1)).float()
        gen_motions_unnorm = self.data_loader.dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        
        gt_motions_renorm_t2m  = self.data_loader.dataset.renorm4t2m(gt_motions.cpu().permute(0, 2, 3, 1)).float()
        gen_motions_renorm_t2m = self.data_loader.dataset.renorm4t2m(sample.cpu().permute(0, 2, 3, 1)).float()

        
        # joints recover
        n_joints = 22
        joints_gt = recover_from_ric(gt_motions_unnorm.squeeze(1), n_joints)
        joints_gen = recover_from_ric(gen_motions_unnorm.squeeze(1), n_joints)


        # t2m motion encoder
        m_lens = lengths
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        gt_motions_renorm_t2m = gt_motions_renorm_t2m[align_idx]
        gen_motions_renorm_t2m = gen_motions_renorm_t2m[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens, 4, rounding_mode="floor")
        
        gen_motion_mov = self.movement_enc(gen_motions_renorm_t2m[..., :-4].squeeze(1)).detach()
        gen_motion_embeddings = self.motion_enc(gen_motion_mov, m_lens)
        gt_motion_mov = self.movement_enc(gt_motions_renorm_t2m[..., :-4].squeeze(1)).detach()
        gt_motion_embeddings = self.motion_enc(gt_motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.text_enc(word_embs.float(), pos_ohot.float(), text_lengths)[align_idx] 

        res = {
                "gt_motions": gt_motions_renorm_t2m,
                "gen_motions": gen_motions_renorm_t2m,
                "text_embeddings": text_emb,
                "gt_motion_embeddings": gt_motion_embeddings,
                "gen_motion_embeddings": gen_motion_embeddings, 
                "joints_ref": joints_gt,
                "joints_gen": joints_gen,
                "our_predicted": logits,
                "label": torch.tensor([self.cassifier_styles.index(gen_style)]*len(lengths)),
                'lengths': lengths,
        }
        
        return res
    
    def _sample(self, texts, lengths, gt_motions):
        batch_size = gt_motions.shape[0]
        n_frames = 196
        
        text_style = [t.replace('.',' in sks style.' ) if '.' in t else t + ' in sks style'  for t in texts]
        
        model_kwargs = {'y':{'text': text_style,
                            'lengths': lengths,
                            'mask': lengths_to_mask(lengths, gt_motions.shape[-1]).unsqueeze(1).unsqueeze(1) 
                            }}
        
        if self.args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(batch_size, device=dist_util.dev()) * self.args.guidance_param
            
        sample_fn = self.diffusion.p_sample_loop
        start = time.time()
        with torch.no_grad():
            sample = sample_fn(
                self.model,
                (batch_size, self.model.njoints, self.model.nfeats, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            
        end = time.time()
        inference_time = torch.tensor(end - start)
        
        return sample, inference_time
    
    def finish(self):
        metrics_dict, count_seq = self.metrics.compute(False)
        print(metrics_dict)
        print(f'num samples={count_seq}')
        return metrics_dict
    
def main():
    args = evaluation_parser()
    args.batch_size = 32

    dist_util.setup_dist(args.device)  
    with torch.no_grad():
        eval = SmoodiEval(args, lora_base_path='save/lora')
        eval._procees_dataset()
        res = eval.finish()  
        
    with open(f"main_eval.json", 'w') as f:
        json.dump(res, f, indent=4)
       
              
if __name__ == '__main__':
    main()