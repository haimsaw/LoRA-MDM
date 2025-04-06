# This code is based on https://github.com/neu-vi/SMooDi/blob/37d5a43b151e0b60c52fc4b37bddbb5923f14bb7/mld/models/metrics/tm2t.py#L72


from typing import List
import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
from scipy.ndimage import uniform_filter1d
import scipy.linalg

class TM2TMetrics(Metric):
    full_state_update = True

    def __init__(self,
                 top_k=3,
                 R_size=32,
                 diversity_times=3,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "matching, fid, and diversity scores"

        self.top_k = top_k
        self.R_size = R_size
        self.diversity_times = diversity_times

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.metrics = []
        # Matching scores
        self.add_state("MM_Dist",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("MM_Dist_gt",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        
        self.Matching_metrics = ["Matching_score", "gt_Matching_score"]
        for k in range(1, top_k + 1):
            self.add_state(
                f"R_precision_top_{str(k)}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.Matching_metrics.append(f"R_precision_top_{str(k)}")
        for k in range(1, top_k + 1):
            self.add_state(
                f"R_precision_top_{str(k)}_gt",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.Matching_metrics.append(f"R_precision_top_{str(k)}_gt")

        self.metrics.extend(self.Matching_metrics)

        # Fid
        self.add_state("FID", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("FID")

        # SRA
        self.add_state("SRA", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("SRA")
        self.add_state("SRA_3", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("SRA_3")
        self.add_state("SRA_5", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("SRA_5")

        #skate_ratio
        self.add_state("skate_ratio", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.metrics.append("skate_ratio")

        # Diversity
        self.add_state("Diversity",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("gt_Diversity",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.metrics.extend(["Diversity", "Diversity_gt"])

        # chached batches
        self.add_state("text_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("genmotion_embeddings", default=[], dist_reduce_fx=None)
        self.add_state("gtmotion_embeddings", default=[], dist_reduce_fx=None)

        self.add_state("predicted_style", default=[], dist_reduce_fx=None)

        self.add_state("labels", default=[], dist_reduce_fx=None)


        self.add_state("joints_rst", default=[], dist_reduce_fx=None)

        self.add_state("lengths", default=[], dist_reduce_fx=None)

        
    def update(
        self,
        text_embeddings: Tensor,
        genmotion_embeddings: Tensor,
        gtmotion_embeddings: Tensor,
        lengths: List[int],
        predicted_style: Tensor,
        label: Tensor,
        joints_rst: Tensor,
    ):
        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # [bs, nlatent*ndim] <= [bs, nlatent, ndim]
        text_embeddings = torch.flatten(text_embeddings, start_dim=1).detach()
        genmotion_embeddings = torch.flatten(genmotion_embeddings,
                                             start_dim=1).detach()
        gtmotion_embeddings = torch.flatten(gtmotion_embeddings,
                                            start_dim=1).detach()
 
        # store all texts and motions
        self.lengths.append(lengths)
        self.text_embeddings.append(text_embeddings)
        self.genmotion_embeddings.append(genmotion_embeddings)
        self.gtmotion_embeddings.append(gtmotion_embeddings)

        
        self.predicted_style.append(predicted_style)

        self.labels.append(label)

        self.joints_rst.append(joints_rst)

        
    
    def compute(self):
        count_seq = self.count_seq.item()

        # init metrics
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # cat all embeddings
        shuffle_idx = torch.randperm(count_seq)
        all_texts = torch.cat(self.text_embeddings,
                              axis=0).cpu()[shuffle_idx, :]
        all_genmotions = torch.cat(self.genmotion_embeddings,
                                   axis=0).cpu()[shuffle_idx, :]
        all_gtmotions = torch.cat(self.gtmotion_embeddings,
                                  axis=0).cpu()[shuffle_idx, :]
        
        all_lengths=self.lengths   
        all_joints_rst = self.joints_rst

        all_predicted = torch.cat(self.predicted_style,
                        axis=0).cpu()[shuffle_idx, :]

        all_labels = torch.cat(self.labels,
                              axis=0).cpu()[shuffle_idx]
        
        output = topk_accuracy(all_predicted, all_labels)
        metrics["SRA"] = output[0]
        metrics["SRA_3"] = output[1]
        metrics["SRA_5"] = output[2]
                    
        # Compute r-precision
        assert count_seq > self.R_size
        top_k_mat = torch.zeros((self.top_k, ))
        for i in range(count_seq // self.R_size):
            # [bs=32, 1*256]
            group_texts = all_texts[i * self.R_size:(i + 1) * self.R_size]
            # [bs=32, 1*256]
            group_motions = all_genmotions[i * self.R_size:(i + 1) *
                                           self.R_size]
            # dist_mat = pairwise_euclidean_distance(group_texts, group_motions)
            # [bs=32, 32]
            dist_mat = euclidean_distance_matrix(group_texts,
                                                 group_motions).nan_to_num()
            # print(dist_mat[:5])
            self.Matching_score += dist_mat.trace()
            argsmax = torch.argsort(dist_mat, dim=1)

            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)

        R_count = count_seq // self.R_size * self.R_size
        metrics["MM_Dist"] = self.Matching_score / R_count
        for k in range(self.top_k):
            metrics[f"R_precision_top_{str(k+1)}"] = top_k_mat[k] / R_count

        # Compute r-precision with gt
        assert count_seq > self.R_size
        top_k_mat = torch.zeros((self.top_k, ))
        for i in range(count_seq // self.R_size):
            # [bs=32, 1*256]
            group_texts = all_texts[i * self.R_size:(i + 1) * self.R_size]
            # [bs=32, 1*256]
            group_motions = all_gtmotions[i * self.R_size:(i + 1) *
                                          self.R_size]
            # [bs=32, 32]
            dist_mat = euclidean_distance_matrix(group_texts,
                                                 group_motions).nan_to_num()
            # match score
            self.gt_Matching_score += dist_mat.trace()
            argsmax = torch.argsort(dist_mat, dim=1)
            top_k_mat += calculate_top_k(argsmax, top_k=self.top_k).sum(axis=0)
        metrics["MM_Dist_gt"] = self.gt_Matching_score / R_count
        for k in range(self.top_k):
            metrics[f"R_precision_top_{str(k+1)}_gt"] = top_k_mat[k] / R_count

        # tensor -> numpy for FID
        all_genmotions = all_genmotions.numpy()
        all_gtmotions = all_gtmotions.numpy()

        # Compute fid
        mu, cov = calculate_activation_statistics_np(all_genmotions)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        metrics["FID"] = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)

        # Compute skate_ratio
        # motions [bs, 22, 3, max_len]
        skate_ratio_sum = 0.0
        for index in range(0,len(all_joints_rst)):
            skate_ratio, skate_vel = calculate_skating_ratio(all_joints_rst[index].permute(0, 2, 3, 1), all_lengths[index])
            skate_ratio_sum += skate_ratio
        
        metrics["skate_ratio"] = sum(skate_ratio_sum) / count_seq

        # Compute diversity

        self.diversity_times = 20
        assert count_seq > self.diversity_times
        metrics["Diversity"] = calculate_diversity_np(all_genmotions,self.diversity_times)
        metrics["Diversity_gt"] = calculate_diversity_np(all_gtmotions, self.diversity_times)
        
        metrics = {k :v.item() if isinstance(v, torch.Tensor) else float(v) for k,v in metrics.items()}
        return metrics, count_seq

def calculate_activation_statistics_np(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

def calculate_diversity_np(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples,
                                     diversity_times,
                                     replace=False)
    second_indices = np.random.choice(num_samples,
                                      diversity_times,
                                      replace=False)
    dist = scipy.linalg.norm(activation[first_indices] -
                             activation[second_indices],
                             axis=1)
    return dist.mean()

def calculate_frechet_distance_np(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (mu1.shape == mu2.shape
            ), "Training and test mean vectors have different lengths"
    assert (sigma1.shape == sigma2.shape
            ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; "
               "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
            # print("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(
        sigma2) - 2 * tr_covmean
    
def euclidean_distance_matrix(matrix1, matrix2):
    """
    Params:
    -- matrix1: N1 x D
    -- matrix2: N2 x D
    Returns:
    -- dist: N1 x N2
    dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    # (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train

    d1 = -2 * torch.mm(matrix1, matrix2.T)  # shape (num_test, num_train)
    d2 = torch.sum(torch.square(matrix1), axis=1,
                   keepdims=True)  # shape (num_test, 1)
    d3 = torch.sum(torch.square(matrix2), axis=1)  # shape (num_train, )
    dists = torch.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = (torch.unsqueeze(torch.arange(size),
                              1).to(mat.device).repeat_interleave(size, 1))
    bool_mat = mat == gt_mat
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        #         print(correct_vec, bool_mat[:, i])
        correct_vec = correct_vec | bool_mat[:, i]
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = torch.cat(top_k_list, dim=1)
    return top_k_mat

def calculate_skating_ratio(motions, lengths):
    # B J 3 T
    thresh_height = 0.05 # 10
    fps = 20.0
    thresh_vel = 0.50 # 20 cm /s 
    avg_window = 5 # frames
    
    mask = lengths_to_mask(lengths,motions.shape[-1]-1)
    batch_size = motions.shape[0]
    # 10 left, 11 right foot. XZ plane, y up
    # motions [bs, 22, 3, max_len]
    verts_feet = motions[:, [10, 11], :, :].detach().cpu().numpy()  # [bs, 2, 3, max_len]
    verts_feet_plane_vel = np.linalg.norm(verts_feet[:, :, [0, 2], 1:] - verts_feet[:, :, [0, 2], :-1],  axis=2) * fps  # [bs, 2, max_len-1]
    # [bs, 2, max_len-1]
    vel_avg = uniform_filter1d(verts_feet_plane_vel, axis=-1, size=avg_window, mode='constant', origin=0)

    verts_feet_height = verts_feet[:, :, 1, :]  # [bs, 2, max_len]
    # If feet touch ground in agjecent frames
    feet_contact = np.logical_and((verts_feet_height[:, :, :-1] < thresh_height), (verts_feet_height[:, :, 1:] < thresh_height))  # [bs, 2, max_len - 1]
    # skate velocity
    skate_vel = feet_contact * vel_avg

    # it must both skating in the current frame
    skating = np.logical_and(feet_contact, (verts_feet_plane_vel > thresh_vel))
    # and also skate in the windows of frames
    skating = np.logical_and(skating, (vel_avg > thresh_vel))


    # Both feet slide
    skating = np.logical_or(skating[:, 0, :], skating[:, 1, :]) # [bs, max_len -1]
    skating = np.logical_and(skating, mask)

    skating_ratio = np.sum(skating, axis=1) / skating.shape[1]
    
    return skating_ratio, skate_vel

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask.cpu().numpy()
    
def topk_accuracy(outputs, labels, topk=(1,3,5)):
    """
    Compute the top-k accuracy for the given outputs and labels.

    :param outputs: Tensor of model outputs, shape [batch_size, num_classes]
    :param labels: Tensor of labels, shape [batch_size]
    :param topk: Tuple of k values for which to compute top-k accuracy
    :return: List of top-k accuracies for each k in topk
    """
    maxk = max(topk)
    
    batch_size = labels.size(0)
    outputs = outputs.squeeze()
    # Get the top maxk indices along the last dimension (num_classes)
    _, pred = outputs.topk(maxk, 1, True, True)

    pred = pred.t()

    # Check if the labels are in the top maxk predictions
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    # Compute accuracy for each k
    accuracies = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        accuracies.append(correct_k.mul_(100.0 / batch_size))
    return accuracies
