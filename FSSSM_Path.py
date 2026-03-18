import copy
import logging
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from options import args_parser
from Model.dde_unet import build_dde_unet
from Dataset.pathology_seg import PathologySegDataset


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, H, W) long
        """
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        union = torch.sum(probs, dims) + torch.sum(targets_one_hot, dims)
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        loss = 1.0 - dice.mean()
        return loss


def build_segmentation_model(args):
    model = build_dde_unet(
        seg_num_classes=args.seg_num_classes,
        base_channels=64,
        num_levels=args.num_entropy_levels,
    )
    return model


def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def cosine_similarity_params(theta_i: torch.Tensor, theta_j: torch.Tensor) -> float:
    return F.cosine_similarity(theta_i.unsqueeze(0), theta_j.unsqueeze(0)).item()


def compute_feature_stats(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    features: (N, D)
    """
    mu = features.mean(dim=0)
    centered = features - mu
    cov = centered.t().mm(centered) / (features.size(0) - 1 + 1e-6)
    return mu, cov


def wasserstein_2_gaussian(mu_i, cov_i, mu_j, cov_j):
    diff = (mu_i - mu_j)
    term1 = torch.sum(diff * diff)
    sqrt_cov_i = torch.linalg.cholesky(cov_i + 1e-6 * torch.eye(cov_i.size(0), device=cov_i.device))
    prod = sqrt_cov_i.mm(cov_j).mm(sqrt_cov_i.t())
    sqrt_prod = torch.linalg.cholesky(prod + 1e-6 * torch.eye(prod.size(0), device=prod.device))
    trace_term = torch.trace(cov_i + cov_j - 2.0 * sqrt_prod)
    return term1 + trace_term


def build_topk_graph(sim_matrix: torch.Tensor, topk: int) -> torch.Tensor:
    """
    sim_matrix: (N, N) similarity or affinity (larger means more similar)
    Returns a row-normalized top-k graph.
    """
    N = sim_matrix.size(0)
    topk = min(topk, N)
    values, indices = torch.topk(sim_matrix, k=topk, dim=1)
    mask = torch.zeros_like(sim_matrix)
    mask.scatter_(1, indices, values)
    mask = mask - torch.diag(torch.diag(mask))  # remove self-loop values
    row_sum = mask.sum(dim=1, keepdim=True) + 1e-6
    graph = mask / row_sum
    return graph


def pseudo_label_diffusion(logits, feat_map, args):
    """
    logits: (B, C, H, W)
    feat_map: (B, C', H, W)
    return refined pseudo labels (B, H, W) long
    """
    with torch.no_grad():
        b, c, h, w = logits.shape
        num_classes = c
        probs = torch.softmax(logits, dim=1)  # (B, C, H, W)
        flat_probs = probs.view(b, c, -1)    # (B, C, N)
        feat = feat_map.view(b, feat_map.size(1), -1)  # (B, C', N)

        refined_labels = []
        for i in range(b):
            f = feat[i]  # (C', N)
            p = flat_probs[i]  # (C, N)
            N_pix = f.size(1)

            # build correlation graph
            sim = torch.matmul(f.t(), f) / (f.size(0) ** 0.5 + 1e-6)  # (N, N)
            C_graph = torch.softmax(sim, dim=1)  # row-wise
            # optional top-k sparsification
            if args.graph_topk > 0 and args.graph_topk < N_pix:
                C_graph = build_topk_graph(C_graph, args.graph_topk)

            y_flat = p  # (C, N)
            y_pix = torch.matmul(y_flat, C_graph)  # (C, N)

            # confidence-based anchors
            conf, pred = torch.max(y_pix, dim=0)  # (N,)
            anchors = (conf > args.tau_conf).nonzero(as_tuple=False).view(-1)
            refined = pred.clone()

            if anchors.numel() > 0:
                for a in anchors:
                    neighbors = (C_graph[a] > args.tau_region).nonzero(as_tuple=False).view(-1)
                    if neighbors.numel() == 0:
                        continue
                    region_scores = y_pix[:, neighbors].sum(dim=1)  # (C,)
                    label = torch.argmax(region_scores)
                    refined[neighbors] = label

            refined_labels.append(refined.view(h, w))

        refined_labels = torch.stack(refined_labels, dim=0)  # (B, H, W)
        return refined_labels


class Client:
    def __init__(self, client_id: int, args, device):
        self.id = client_id
        self.args = args
        self.device = device
        self.model = build_segmentation_model(args).to(device)
        # anchor head input dim = global feature dim
        feat_dim = 64 + args.num_entropy_levels
        self.anchor_head = nn.Linear(feat_dim, args.seg_num_classes).to(device)
        # AdamW optimizer as in implementation details
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.anchor_head.parameters()),
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )
        self.dice_loss = DiceLoss()

        self.mu = None
        self.cov = None
        self.anchor_features = None

    def local_train(self, loader_labeled: DataLoader, loader_unlabeled: DataLoader,
                    anchor_bank: torch.Tensor, all_anchor_preds: List[torch.Tensor],
                    neighbor_indices: List[int]):
        self.model.train()

        ce_loss_fn = nn.CrossEntropyLoss()

        if anchor_bank is not None and anchor_bank.numel() > 0 and len(neighbor_indices) > 0:
            # subsample anchors for efficiency
            num_anchors = min(self.args.num_anchor_features, anchor_bank.size(0))
            idx = torch.randperm(anchor_bank.size(0))[:num_anchors]
            anchors = anchor_bank[idx].to(self.device)  # (A, D)
            p_k = torch.softmax(self.anchor_head(anchors), dim=1)  # (A, C)
        else:
            anchors = None
            p_k = None

        for (x_l, y_l), (x_u, _) in zip(loader_labeled, loader_unlabeled):
            x_l = x_l.to(self.device)
            y_l = y_l.to(self.device).long()
            x_u = x_u.to(self.device)

            logits_sup, feat_sup, _ = self.model(x_l)
            loss_ce = ce_loss_fn(logits_sup, y_l)
            loss_dice = self.dice_loss(logits_sup, y_l)
            loss_sup = loss_ce + loss_dice

            logits_u, feat_u, global_feat_u = self.model(x_u)
            pseudo_labels = pseudo_label_diffusion(logits_u.detach(), feat_u.detach(), self.args)
            loss_pseudo = ce_loss_fn(logits_u, pseudo_labels.to(self.device))

            loss_consistency = torch.tensor(0.0, device=self.device)
            if anchors is not None and p_k is not None and len(neighbor_indices) > 0 and all_anchor_preds:
                neighbor_losses = []
                for j in neighbor_indices:
                    p_j = all_anchor_preds[j].to(self.device)  # (A, C)
                    A = min(p_k.size(0), p_j.size(0))
                    if A == 0:
                        continue
                    diff = p_k[:A] - p_j[:A]
                    neighbor_losses.append(torch.mean(diff * diff))
                if neighbor_losses:
                    loss_consistency = torch.stack(neighbor_losses).mean()

            loss = loss_sup + self.args.lambda_pseudo * loss_pseudo + self.args.lambda_consistency * loss_consistency

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.update_feature_stats(loader_unlabeled)

    def update_feature_stats(self, loader_unlabeled: DataLoader):
        self.model.eval()
        feats = []
        with torch.no_grad():
            for x_u, _ in loader_unlabeled:
                x_u = x_u.to(self.device)
                _, _, global_feat = self.model(x_u)
                feats.append(global_feat.cpu())
        if len(feats) > 0:
            feats = torch.cat(feats, dim=0)
            mu, cov = compute_feature_stats(feats)
            self.mu = mu
            self.cov = cov
            # sample anchors from global features
            num_anchors = min(self.args.num_anchor_features, feats.size(0))
            idx = torch.randperm(feats.size(0))[:num_anchors]
            self.anchor_features = feats[idx]


def compute_dual_neighbors(clients: List[Client], args, device):
    """Compute dual-domain distances and trusted neighbor sets."""
    num_clients = len(clients)

    thetas = [flatten_params(c.model).to(device) for c in clients]
    mus = [c.mu.to(device) for c in clients]
    covs = [c.cov.to(device) for c in clients]

    with torch.no_grad():
        theta_mat = torch.stack(thetas, dim=0)  # (N, P)

        norm = torch.norm(theta_mat, dim=1, keepdim=True) + 1e-8
        theta_norm = theta_mat / norm
        sim_matrix = torch.mm(theta_norm, theta_norm.t())  # (N, N)

        D = torch.zeros(num_clients, num_clients, device=device)
        for i in range(num_clients):
            for j in range(num_clients):
                if i == j:
                    D[i, j] = 0.0
                    continue
                E_ij = torch.norm(theta_mat[i] - theta_mat[j], p=2)
                W_ij = wasserstein_2_gaussian(mus[i], covs[i], mus[j], covs[j])
                D_ij = W_ij + args.dual_alpha * E_ij
                D[i, j] = torch.clamp(D_ij, max=args.dual_C1)

        neighbor_lists: List[List[int]] = []
        for k in range(num_clients):
            neighbors = []
            for j in range(num_clients):
                if k == j:
                    continue
                if D[k, j].item() < args.tau_trust and sim_matrix[k, j].item() > args.tau_sim:
                    neighbors.append(j)
            neighbor_lists.append(neighbors)

    return neighbor_lists, D, sim_matrix


def decentralized_aggregation(clients: List[Client], args, device):
    """Decentralized aggregation using trusted neighbors."""
    neighbor_lists, D, sim_matrix = compute_dual_neighbors(clients, args, device)
    num_clients = len(clients)

    thetas = [flatten_params(c.model).to(device) for c in clients]
    theta_mat = torch.stack(thetas, dim=0)

    with torch.no_grad():
        for k in range(num_clients):
            neighbors = neighbor_lists[k]
            if not neighbors:
                continue

            d_vals = D[k, neighbors]
            weights = torch.softmax(-d_vals, dim=0)  # (len(neighbors),)
            new_theta = torch.zeros_like(theta_mat[k])
            for w, j in zip(weights, neighbors):
                new_theta += w * theta_mat[j]

            # load back to model parameters
            offset = 0
            for p in clients[k].model.parameters():
                numel = p.data.numel()
                p.data.copy_(new_theta[offset:offset + numel].view_as(p.data))
                offset += numel


def evaluate_on_validation(clients: List[Client], val_loader: DataLoader, num_classes: int, device):
    """Evaluate mean Dice and mIoU on validation set."""
    eps = 1e-6
    client_dice_scores = []
    client_iou_scores = []

    for client in clients:
        client.model.eval()
        intersect_sum = torch.zeros(num_classes, device=device)
        union_sum = torch.zeros(num_classes, device=device)
        dice_inter = torch.zeros(num_classes, device=device)
        dice_union = torch.zeros(num_classes, device=device)

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device).long()
                logits, _, _ = client.model(x)
                preds = torch.argmax(logits, dim=1)  # (B,H,W)

                for c in range(num_classes):
                    pred_c = (preds == c)
                    target_c = (y == c)
                    inter = (pred_c & target_c).sum()
                    union = (pred_c | target_c).sum()
                    dice_inter[c] += 2 * inter
                    dice_union[c] += (pred_c.sum() + target_c.sum())
                    intersect_sum[c] += inter
                    union_sum[c] += union

        dice_per_class = (dice_inter + eps) / (dice_union + eps)
        iou_per_class = (intersect_sum + eps) / (union_sum + eps)

        client_dice_scores.append(dice_per_class.mean().item())
        client_iou_scores.append(iou_per_class.mean().item())

    mean_dice = float(np.mean(client_dice_scores)) if client_dice_scores else 0.0
    mean_miou = float(np.mean(client_iou_scores)) if client_iou_scores else 0.0
    return mean_dice, mean_miou


def main():
    args = args_parser()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    log_dir = f'./results/FSSSM_Path/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'FSSSM_Path.log')

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    full_dataset = PathologySegDataset(
        root=args.path_pathology_seg,
        patch_size=args.patch_size,
        labeled=True,
    )

    num_samples = len(full_dataset)
    indices = np.arange(num_samples)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(indices)

    num_val = int(args.val_ratio * num_samples)
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    val_dataset = PathologySegDataset(
        root=args.path_pathology_seg,
        indices=val_indices,
        patch_size=args.patch_size,
        labeled=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.seg_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    num_clients = args.num_clients
    client_indices = np.array_split(train_indices, num_clients)

    clients = []

    labeled_ratio = args.seg_labeled_ratio
    for cid in range(num_clients):
        idxs = client_indices[cid]
        n_total = len(idxs)
        n_labeled = max(1, int(labeled_ratio * n_total))
        labeled_idx = idxs[:n_labeled]
        unlabeled_idx = idxs[n_labeled:]

        client = Client(cid, args, device)
        client.labeled_idx = labeled_idx
        client.unlabeled_idx = unlabeled_idx
        clients.append(client)

    num_rounds = args.seg_num_rounds
    batch_size = args.seg_batch_size

    for r in range(1, num_rounds + 1):
        logging.info(f"Round {r} / {num_rounds}")

        if r == 1 or any(c.mu is None for c in clients):
            neighbor_lists = [[] for _ in range(num_clients)]
            anchor_bank = None
            all_anchor_preds: List[torch.Tensor] = []
        else:
            neighbor_lists, _, _ = compute_dual_neighbors(clients, args, device)
            # build global anchor feature pool
            anchor_list = [c.anchor_features for c in clients if c.anchor_features is not None]
            if anchor_list:
                anchor_bank = torch.cat(anchor_list, dim=0)  # (A_total, D)
                # precompute neighbors' predictions on anchors (no grad)
                all_anchor_preds = []
                with torch.no_grad():
                    for c in clients:
                        if c.anchor_features is None:
                            all_anchor_preds.append(torch.empty(0, args.seg_num_classes))
                        else:
                            p = torch.softmax(
                                c.anchor_head(anchor_bank.to(c.device)), dim=1
                            ).cpu()
                            all_anchor_preds.append(p)
            else:
                anchor_bank = None
                all_anchor_preds = [torch.empty(0, args.seg_num_classes) for _ in clients]

        for client_idx, client in enumerate(clients):
            labeled_dataset = PathologySegDataset(
                root=args.path_pathology_seg,
                indices=client.labeled_idx,
                patch_size=args.patch_size,
                labeled=True,
            )
            unlabeled_dataset = PathologySegDataset(
                root=args.path_pathology_seg,
                indices=client.unlabeled_idx,
                patch_size=args.patch_size,
                labeled=False,
            )

            loader_labeled = DataLoader(
                labeled_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
            loader_unlabeled = DataLoader(
                unlabeled_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )

            client.local_train(
                loader_labeled,
                loader_unlabeled,
                anchor_bank=anchor_bank,
                all_anchor_preds=all_anchor_preds,
                neighbor_indices=neighbor_lists[client_idx],
            )

        decentralized_aggregation(clients, args, device)

        mean_dice, mean_miou = evaluate_on_validation(clients, val_loader, args.seg_num_classes, device)
        logging.info(f"Round {r}: Val Dice={mean_dice:.4f}, mIoU={mean_miou:.4f}")
        print(f"[Round {r}/{num_rounds}] Val Dice={mean_dice:.4f}, mIoU={mean_miou:.4f}")

    logging.info("FSSSM-Path training finished.")


if __name__ == '__main__':
    main()

