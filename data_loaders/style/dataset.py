# This code is based on https://github.com/neu-vi/SMooDi/blob/37d5a43b151e0b60c52fc4b37bddbb5923f14bb7/mld/data/humanml/data/dataset.py#L124

import numpy as np
import os
import codecs as cs
import random
from rich.progress import track

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from os.path import join as pjoin

def build_dict_from_txt(filename):
    result_dict = {}
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) >= 3:
                key = parts[0]
                value = parts[2].split("_")[0]
                value2 = parts[1].split("_")[0]
                value3 = parts[1].split("_")[1]
                value4 = parts[1].split("_")[2].split('.')[0]

                result_dict[key] = value, value2, value3, value4
                
    return result_dict
 
class StyleMotionDataset(Dataset):

    def __init__(self, styles, mode, motion_type_to_exclude=[]):
        assert styles is not None
        
        data_dict = {}
        id_list = []
        self.max_length = 20
        self.max_motion_length = 196
        self.unit_length = 4
        self.pointer = 0
        self.styles = styles
        self.mode = mode
        
        if mode == 'train':
            self.tokens=['sks', 'hta', 'oue', 'asar', 'nips']
            assert len(self.styles) <= len(self.tokens)
        
        path = "./dataset/100STYLE-SMPL/"
        split_file = path + "train_100STYLE_Full.txt" 
                
        self.movments = {
           "BR":	"running backwards",
            "BW":	"walking backwards",
            "FR":	"running",
            "FW":	"walking",
            "ID":	"standing",
            "SR":	"running sideways",
            "SW":	"walking sideways", 
        }

        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        dict_path = path + "100STYLE_name_dict.txt"
        motion_to_label = build_dict_from_txt(dict_path)
        
        mean = np.load(path + "/Mean.npy")
        std = np.load(path+ "/Std.npy")
        new_name_list = []
        length_list = []
        label_list = []
        count = 0
        bad_count = 0
        new_name_list = []

        motion_dir = path + "/new_joint_vecs-001"

        text_dir = path + "/texts"
        length_list = []

        enumerator = enumerate(
            track(
                id_list,
                f"Loading 100STYLE {len(styles)} {split_file.split('/')[-1].split('.')[0]}"
            ))
        maxdata = 1e10 

        self.min_motion_length = 40

        for i, name in enumerator:
            if count > maxdata:
                break
            
            label_data, motion_style, motion_type, cut_idx = motion_to_label[name]
            if motion_style not in  styles:
                continue
            if motion_type.startswith("TR"):
                continue
            if motion_type in motion_type_to_exclude:
                continue
            
            motion = np.load(pjoin(motion_dir, name + ".npy"))

            if (len(motion)) < self.min_motion_length:
                continue
            text_data = []
            
            text_path = pjoin(text_dir, name + ".txt")
            assert os.path.exists(text_path)
            with cs.open(text_path) as f:
                for line in f.readlines():
                    text_dict_2 = {}
                    line_split = line.strip().split("#")
                    caption = line_split[0]
                    text_dict_2["caption"] = caption
                    text_data.append(text_dict_2)
            
            data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "label": label_data,
                        "style": motion_style,
                        "text": text_data,
                        "motion_type": motion_type, 
                        "cut_idx": cut_idx,
                        "token": self.tokens[self.styles.index(motion_style)] if mode == 'train' else 'sks'
                    }
 
            new_name_list.append(name)
            length_list.append(len(motion))
            count += 1
        
        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
    
    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform(self,data):
        return (data - self.mean) / self.std

    def get_mean_std(self):
        return self.mean, self.std
    
    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, label, text_list, style, motion_type, cut_idx, style_token = data["motion"], data["length"], data["label"], data["text"], data["style"], data["motion_type"], data["cut_idx"], data["token"]

        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length

        movement = self.movments[motion_type]
        caption = f'A person is {movement} in {style_token} style.'

        m_length = min(196, m_length)
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        return {
            "inp": torch.tensor(motion.T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
            "action":torch.tensor(int(label)),
            "lengths":m_length,
            "text":caption,
            "style": style,
            "action_text": motion_type,
            "cut_idx": cut_idx,
        }
        

def random_zero_out(data, percentage=0.4, probability=0.6,noise_probability=0.8,noise_level=0.05):
    if random.random() < probability:
        # Calculate the total number of sequences to zero out
        num_sequences = data.shape[0]

        percentage = np.random.rand() * 0.5
        num_to_zero_out = int(num_sequences * percentage)

        # Randomly choose sequence indices to zero out
        indices_to_zero_out = np.random.choice(num_sequences, num_to_zero_out, replace=False)

        # Zero out the chosen sequences
        data[indices_to_zero_out, :] = 0

    # data = shuffle_segments_numpy(data,16)

    if random.random() < noise_probability:
        noise = np.random.normal(0, noise_level * np.ptp(data), data.shape)
        data += noise

    return data


