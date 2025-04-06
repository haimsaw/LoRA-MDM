from argparse import ArgumentParser
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import wandb
from data_loaders.get_data import get_dataset_loader
import json
from utils import dist_util


normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"  # for debug

class MotionDiscriminatorConv(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, device, output_size=12, use_noise=None):
        super(MotionDiscriminatorConv, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.use_noise = use_noise
        self.output_size = output_size

        self.conv = nn.Sequential(
            nn.Conv1d(self.input_size, self.hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=1),  # First downsample
            *[nn.Sequential(
                nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
                nn.ReLU()
            ) for _ in range(self.hidden_layer)],
            nn.AvgPool1d(kernel_size=4, stride=2, padding=1),  # Downsizing in the middle
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, motion_sequence, lengths=None, hidden_unit=None):
        bs, njoints, nfeats, num_frames = motion_sequence.shape
        motion_sequence = motion_sequence.reshape(bs, njoints*nfeats, num_frames)  # (N,C,L)
        
        x = self.conv(motion_sequence)  # Apply convolutions and downsizing
        
        # Create mask from lengths
        if lengths is not None:
            # Adjust lengths for the downsampled sequence
            downsampled_lengths = torch.ceil((lengths.float() / 4) * 0.75).long()  # Account for 2 pooling layers with stride 2  # Ignore last 25% of the sequence to avoid overfitting on lengths
            mask = torch.arange(x.size(2), device=x.device)[None, :] < downsampled_lengths[:, None]
            mask = mask.unsqueeze(1).float()  # Add channel dimension
            # Masked average pooling
            x = (x * mask).sum(dim=2) / downsampled_lengths.unsqueeze(1).float()
        else:
            x = x.mean(dim=2)  # Global average pooling across time dimension
            
        x = self.linear(x)  # Final linear layer
        return x

charcter_classes = ["Aeroplane", "Cat", "Chicken", "Dinosaur", "FairySteps", "Monk", "Morris", "Penguin", "Quail", "Roadrunner", "Robot", "Rocket", "Star", "Superman"]  # 14 excluding Zombie
action_classes = ["Akimbo", "ArmsAboveHead", "ArmsBehindBack", "ArmsBySide", "ArmsFolded", "BeatChest", "BentForward", "BentKnees", 
                  "BigSteps", "BouncyLeft", "BouncyRight", "CrossOver", "FlickLegs", "Followed", "GracefulArms", "HandsBetweenLegs", 
                  "HandsInPockets", "HighKnees", "KarateChop", "Kick", "LeanBack", "LeanLeft", "LeanRight", "LeftHop", "LegsApart",
                  "LimpLeft", "LimpRight", "LookUp", "Lunge", "March", "Punch", "RaisedLeftArm", "RaisedRightArm", "RightHop", "Skip",
                  "SlideFeet", "SpinAntiClock", "SpinClock", "StartStop", "Strutting", "Sweep", "Teapot", "Tiptoe", "TogetherStep",
                  "TwoFootJump", "WalkingStickLeft", "WalkingStickRight", "Waving"]
badly_proccesd_styles = ['Zombie','WiggleHips', 'WhirlArms', 'WildArms', 'WildLegs', 'WideLegs']

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--exp_name", default="debug", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=1e-6, type=float)
    parser.add_argument("--n_epoches", default=500, type=int)
    parser.add_argument("--disc_dim",type=int, default=64)
    parser.add_argument("--n_hidden_layers",type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=10, help="Interval for evaluation")
    parser.add_argument("--save_interval", type=int, default=100, help="Interval for saving checkpoints")
    parser.add_argument("--style_group", type=str, default='No Action', choices=['All', 'No Action','Character'])
    parser.add_argument("--input_size", type=int, default=263, help="Input size for classifier")

    args = parser.parse_args()
    args.save_path = f"./save/classifier/{args.exp_name}"
    args.output_size = 46
    if args.style_group == 'All':
        args.output_size = 100 - len(badly_proccesd_styles)
    elif args.style_group == 'Charcter':
        args.output_size = len(charcter_classes)
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    return args

def get_style_mappings(style_group='All'):
    """
    Get style mappings for both SMooDi and our dataset
    Args:
        classes_type: 'all' or 'charcter' to filter only character classes
    Returns:
        tuple: (styles_key, label_to_style_smoodi, label_to_style_us, style_to_label_smoodi, style_to_label_us)
    """
    label_to_style = {}
        
    # get style info - ours's version 
    with open('dataset/100STYLE-SMPL/100STYLE_name_dict.txt', 'r') as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) >= 3:
                key = int(parts[2])
                value = parts[1].split("_")[0]
                if value in badly_proccesd_styles:
                    continue
                label_to_style[key] = value
    style_to_label = {v:k for k,v in label_to_style.items()}
    
    # sort styles by label to make the labels consistent
    styles_key = sorted(list(style_to_label.keys()))
    
    # Filter styles if we are using character classes
    if style_group == 'Charcter':
        styles_key = [s for s in styles_key if s in charcter_classes]
    elif style_group == 'No Action':
        styles_key = [s for s in styles_key if s not in action_classes]
        
    return styles_key, label_to_style, style_to_label

def main():
    args = get_args()
    
    wandb.init(
        project='LoRA-MDM',
        name='cls_'+args.exp_name,
        config=args,
    )
    os.makedirs(args.save_path, exist_ok=True)

    device = args.device
    
    # Get style mappings
    styles_key, _, style_to_label = get_style_mappings(args.classes_type)
    
    labels_key_us = [style_to_label[s] for s in styles_key]
    
    print(f'styles_key:{styles_key}')
    # print(f'labels_key_smoodi:{labels_key_smoodi}')
    print(f'labels_key_us:{labels_key_us}')
    
    # Save args to JSON file after we know the output size
    args_dict = vars(args)
    with open(os.path.join(args.save_path, 'args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    style_loader_train = get_dataset_loader(name='100style', styles=tuple(styles_key), batch_size=args.batch_size,
                                            num_frames=None, split='train_calssifir', motion_type_to_exclude=("BR",) )
    style_loader_test = get_dataset_loader(name='100style', styles=tuple(styles_key), batch_size=args.batch_size
                                           , num_frames=None, split='train_calssifir', motion_type_to_exclude=("BW","FR", "FW", "ID", "SR", "SW"))
    
    
    classifier = MotionDiscriminatorConv(input_size=args.input_size, 
                                      hidden_size=args.disc_dim, 
                                      hidden_layer=args.n_hidden_layers, 
                                      device=device, 
                                      output_size=args.output_size).to(device)
    wandb.watch(classifier, log='all', log_freq=100)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    
    # Add tracking variables for best model
    best_hit5 = 0
    best_epoch = 0
    
    for epoch in tqdm(range(args.n_epoches)):
        running_loss = 0
        hit1, hit3, hit5 = 0,0,0
        for i, (motions, kwargs) in enumerate(style_loader_train):
            
            # initilize step
            optimizer.zero_grad()
            target = torch.tensor([labels_key_us.index(v) for v in kwargs['y']['action']], device=device)
            motions = motions.to(device)
            lengths = kwargs['y']['lengths'].to(device)

            # forward pass
            outputs = classifier(motions, lengths)
            loss = loss_fn(outputs, target)
            
            # backward pass
            loss.backward()
            optimizer.step()

            top = outputs.topk(k=5, dim=1)[1]==target.unsqueeze(1)
            hit1 += top[...,:1].sum().item()
            hit3 +=  top[...,:3].sum().item()
            hit5 += top[...,:5].sum().item()

            running_loss += loss.item() * motions.shape[0]
            
        wandb.log({"loss": running_loss/len(style_loader_train.dataset),
                    "acc1":hit1/len(style_loader_train.dataset),
                    "acc3":hit3/len(style_loader_train.dataset),
                    "acc5":hit5/len(style_loader_train.dataset)},
                  step=epoch)
            
        if epoch % args.eval_interval == 0:
            print(f'=== START EVALUATION FOR EPOCH {epoch} ===')
            # evaluate on test set, simple prompts and humanml prompts
            eval_test(device, labels_key_us, style_loader_test, classifier, loss_fn, epoch)
            hit5_acc = eval_my_gen(device, styles_key, classifier, loss_fn, epoch, file='humanml_test', dataloader=style_loader_test, eval_dir='generated/us_1000_steps', eval_name='us_hml_test')
            
            # Update best model if current hit5 accuracy is better
            if hit5_acc > best_hit5:
                best_hit5 = hit5_acc
                best_epoch = epoch
                # Save best model
                torch.save(classifier.state_dict(), args.save_path+f'/best.pt')
                
                # Update tracking file
                with open(os.path.join(args.save_path, 'model_info.txt'), 'w') as f:
                    f.write(f'Best model epoch: {best_epoch} (hit5: {best_hit5:.3f})\n')
                    f.write(f'Current epoch: {epoch}\n')

    # Save final model
    torch.save(classifier.state_dict(), args.save_path+f'/classifier_last.pt')
    
    # Update tracking file with final information
    with open(os.path.join(args.save_path, 'model_info.txt'), 'w') as f:
        f.write(f'Best model epoch: {best_epoch} (hit5: {best_hit5:.3f})\n')
        f.write(f'Last model epoch: {epoch}\n')
    
    wandb.finish()
    
@torch.no_grad()
def eval_test(device, labels_key_us, style_loader_test, classifier, loss_fn, epoch):
    running_loss = 0
    hit1, hit3, hit5 = 0,0,0
    for j, (motions, kwargs) in enumerate(style_loader_test):
        target = torch.tensor([labels_key_us.index(v) for v in kwargs['y']['action']], device=device)
        motions = motions.to(device)
        lengths = kwargs['y']['lengths'].to(device)
            
        outputs = classifier(motions, lengths)
        loss = loss_fn(outputs, target)
                
        top = outputs.topk(k=5, dim=1)[1]==target.unsqueeze(1)
        hit1 += top[...,:1].sum().item()
        hit3 +=  top[...,:3].sum().item()
        hit5 += top[...,:5].sum().item()

        running_loss += loss.item()*motions.shape[0]
                
    wandb.log({"loss_eval": running_loss/len(style_loader_test.dataset),
                        "acc1_eval":hit1/len(style_loader_test.dataset),
                        "acc3_eval":hit3/len(style_loader_test.dataset),
                        "acc5_eval":hit5/len(style_loader_test.dataset)},
                        step=epoch)

    # Log aggregate metrics
    print(f"Evaluation metrics for test set:")
    print(f"Accuracy@5: {hit5 / len(style_loader_test.dataset):.3f}")
            
def eval_my_gen(device, styles_key, classifier, loss_fn, epoch, file, dataloader, eval_dir='generated/us_1000_steps', eval_name='us_hml_test'):
    total_running_loss = 0
    total_hit1, total_hit3, total_hit5 = 0, 0, 0
    total_samples = 0
    
    for style in tqdm(styles_key, desc=f'Evaluating {eval_name}'):
        # Load and evaluate each style
        try:
            # Find the matching style directory by ignoring suffixes after _
            style_dirs = [d for d in os.listdir(eval_dir) if d.split('_')[0] == style]
            if not style_dirs:
                raise FileNotFoundError(f"No directory found for style {style}")
            style_dir = style_dirs[0]  # Take the first matching directory
            data_path = os.path.join(eval_dir, style_dir, file, 'results.npy')
            data = np.load(data_path, allow_pickle=True)[()]
            all_lengths = data['lengths']
            all_motions = data['rics']  # B 1 T J
            bs = len(all_motions)
            total_samples += bs

            motions = dataloader.dataset.transform(torch.from_numpy(all_motions)).permute(0, 3, 1, 2).to(device)
            lengths = torch.from_numpy(all_lengths).to(device)
            target = torch.tensor([styles_key.index(style)], device=device).repeat(bs)

            outputs = classifier(motions, lengths)
            loss = loss_fn(outputs, target)
            
            top = outputs.topk(k=5, dim=1)[1]==target.unsqueeze(1)
            total_hit1 += top[...,:1].sum().item()
            total_hit3 += top[...,:3].sum().item()
            total_hit5 += top[...,:5].sum().item()

            total_running_loss += loss.item() * bs
            
            
        except FileNotFoundError:
            print(f"Warning: No data found for style {style}")
            continue
    
    # Skip if no data was processed
    if total_samples == 0:
        print("Warning: No data was processed in eval_my_gen")
        return 0.0
        
    # Log aggregate metrics
    print(f"Evaluation metrics for {eval_name}:")
    print(f"Accuracy@5: {total_hit5 / total_samples:.3f}")
    
    wandb.log({
        f"loss_eval_all_{eval_name}": total_running_loss / total_samples,
        f"acc1_eval_all_{eval_name}": total_hit1 / total_samples,
        f"acc3_eval_all_{eval_name}": total_hit3 / total_samples,
        f"acc5_eval_all_{eval_name}": total_hit5 / total_samples
    }, step=epoch)
    
    return total_hit1/ total_samples, total_hit3/ total_samples , total_hit5 /total_samples  # Return average hit5 accuracy across all styles
        
def load_classifier(style_group):
    """
    Load a trained classifier from a directory
    Args:
        classifier_dir: Path to directory containing args.json, best.pt and model_info.txt
    Returns:
        tuple: (classifier, styles_key, label_to_style_smoodi, label_to_style_us)
    """
    
    
    if style_group == 'No Action':
        classifier_dir = 'save/Classifier_NoAction' 
    elif style_group == 'All':
        classifier_dir = 'save/Classifier_All' 
    elif style_group == 'Character':
        classifier_dir = 'save/Classifier_Charcter'
    
    # Load args
    with open(os.path.join(classifier_dir, 'args.json'), 'r') as f:
        args = json.load(f)
        print("\nLoaded args:")
        print(json.dumps(args, indent=2))
    
    # Load model info
    with open(os.path.join(classifier_dir, 'model_info.txt'), 'r') as f:
        info = f.readlines()
        best_epoch = int(info[0].split(':')[1].split('(')[0])
        print(f"\nLoading best model from epoch {best_epoch}")
    
    # Get style mappings
    styles_key, label_to_style, _ = get_style_mappings(style_group)
    
    # Create classifier instance
    device = dist_util.dev()
    classifier = MotionDiscriminatorConv(
        input_size=args['input_size'],
        hidden_size=args['disc_dim'],
        hidden_layer=args['n_hidden_layers'],
        device=device,
        output_size=args['output_size']
    )
    
    # Load weights
    classifier.load_state_dict(torch.load(os.path.join(classifier_dir, 'best.pt')))
    classifier = classifier.to(device)
    classifier.eval()
    
    return classifier, styles_key, label_to_style

if __name__== '__main__':
    main()