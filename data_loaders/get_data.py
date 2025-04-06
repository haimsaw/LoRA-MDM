import torch
from torch.utils.data import DataLoader
from data_loaders.humanml_utils import get_inpainting_mask
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    elif name == "100style":
        from data_loaders.style.dataset import StyleMotionDataset
        return StyleMotionDataset
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', styles=None, motion_type_to_exclude=[]):
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode)
    elif name == "100style":
        dataset = DATA(styles, split,motion_type_to_exclude=motion_type_to_exclude)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset

def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', styles=None, debug=False, motion_type_to_exclude=()):
    dataset = get_dataset(name, num_frames, split, hml_mode, styles, motion_type_to_exclude)
    collate = get_collate_fn(name, hml_mode)
    batch_size = 5 if debug else min(len(dataset), batch_size)
    drop_last = name == "humanml"
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, drop_last=drop_last, collate_fn=collate, pin_memory=True
    )
    return loader

def get_prior_dataset_loader(batch_size, num_frames):
    from data_loaders.humanml.data.dataset import HumanML3D
    data = HumanML3D(split='train', num_frames=num_frames, mode='train')
    collate = get_collate_fn('humanml', 'train')
   
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=True,
        num_workers=2,
        drop_last=True, pin_memory=True, collate_fn=collate
        )
    return loader

    
class InpaintingDataLoader(object):
    def __init__(self, data, inpainting_mask):
        self.data = data
        self.inpainting_mask = inpainting_mask
    
    def __iter__(self):
        for motion, cond in super().__getattribute__('data').__iter__():
            cond['y']['inpainting_mask'] = torch.tensor(get_inpainting_mask(self.inpainting_mask, motion.shape)).to(motion.device)
            yield motion, cond
    
    def __getattribute__(self, name):
        return super().__getattribute__('data').__getattribute__(name)
    
    def __len__(self):
        return len(super().__getattribute__('data'))