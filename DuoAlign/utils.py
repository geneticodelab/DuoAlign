import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from .preprocess import pca
import matplotlib.pyplot as plt


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()

    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)

    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def clustering(adata, n_clusters=7, key='emb', add_key='SpatialGlue', method='mclust', start=0.1, end=3.0,
               increment=0.01, use_pca=False, n_comps=20):

    if use_pca:
        adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)

    if method == 'mclust':
        if use_pca:
            adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters)
        else:
            adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
        adata.obs[add_key] = adata.obs['mclust']

    elif method == 'leiden':
        use_rep = key + '_pca' if use_pca else key
        res = search_res(adata, n_clusters, use_rep=use_rep, method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs['leiden']

    elif method == 'louvain':
        use_rep = key + '_pca' if use_pca else key
        res = search_res(adata, n_clusters, use_rep=use_rep, method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs['louvain']


def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    print('Searching resolution...')
    label = 0

    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)

    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))

        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))

        if count_unique == n_clusters:
            label = 1
            break

    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."

    return res


def plot_weight_value(alpha, label, modality1='mRNA', modality2='protein'):
    import pandas as pd

    df = pd.DataFrame(columns=[modality1, modality2, 'label'])
    df[modality1], df[modality2] = alpha[:, 0], alpha[:, 1]
    df['label'] = label

    df = df.set_index('label').stack().reset_index()
    df.columns = ['label_SpatialGlue', 'Modality', 'Weight value']

    ax = sns.violinplot(data=df, x='label_SpatialGlue', y='Weight value', hue="Modality",
                        split=True, inner="quart", linewidth=1, show=False)
    ax.set_title(modality1 + ' vs ' + modality2)

    plt.tight_layout(w_pad=0.05)
    plt.show()


def evaluate_clustering(y_true, y_pred):
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import (
        adjusted_rand_score, normalized_mutual_info_score,
        mutual_info_score, adjusted_mutual_info_score,
        fowlkes_mallows_score, v_measure_score,
        f1_score, jaccard_score, completeness_score
    )
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    mi = mutual_info_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    fmi = fowlkes_mallows_score(y_true, y_pred)
    v_measure = v_measure_score(y_true, y_pred)
    compl = completeness_score(y_true, y_pred)

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind_row, ind_col = linear_sum_assignment(w.max() - w)
    map_dict = {row: col for row, col in zip(ind_row, ind_col)}
    y_mapped = np.array([map_dict.get(p, p) for p in y_pred])

    acc = np.sum(y_mapped == y_true) / y_true.size
    f1 = f1_score(y_true, y_mapped, average='micro')
    jaccard = jaccard_score(y_true, y_mapped, average='micro')

    return {
        'ACC': acc, 'ARI': ari, 'NMI': nmi, 'MI': mi,
        'AMI': ami, 'FMI': fmi, 'V-Measure': v_measure,
        'F1': f1, 'Jaccard': jaccard, 'Compl': compl
    }


def sinkhorn_knopp(affinity, eps=0.05, iters=50, nu=None):
    N, M = affinity.shape
    if nu is None:
        nu = torch.ones(M, device=affinity.device) / M

    if not torch.isfinite(affinity).all():
        affinity = torch.nan_to_num(affinity, nan=0.0, posinf=0.0, neginf=0.0)

    stable = affinity - affinity.max(dim=1, keepdim=True).values
    Q = torch.exp(stable / eps)
    Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-12)

    for _ in range(iters):
        Q = Q * (nu * N / (Q.sum(dim=0, keepdim=True) + 1e-12))
        Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-12)
    return Q


def psa_loss(z_spat, z_feat):
    return F.mse_loss(z_spat, z_feat)


def prototype_contrastive_loss(z_a, z_b, protos, tau=0.1, eps=0.05, iters=50,
                               nu=None, detach_target=True,
                               use_sinkhorn=True, symmetric=True):
    z_a_n = F.normalize(z_a, p=2, dim=1)
    z_b_n = F.normalize(z_b, p=2, dim=1)
    protos_n = F.normalize(protos, p=2, dim=1)

    logits_a = torch.matmul(z_a_n, protos_n.t()) / tau
    logits_b = torch.matmul(z_b_n, protos_n.t()) / tau
    logits_a = torch.nan_to_num(logits_a, nan=0.0, posinf=0.0, neginf=0.0)
    logits_b = torch.nan_to_num(logits_b, nan=0.0, posinf=0.0, neginf=0.0)

    if use_sinkhorn:
        q_a = sinkhorn_knopp(logits_a, eps=eps, iters=iters, nu=nu)
        q_b = sinkhorn_knopp(logits_b, eps=eps, iters=iters, nu=nu)
    else:
        q_a = F.softmax(logits_a, dim=1)
        q_b = F.softmax(logits_b, dim=1)

    if detach_target:
        q_a = q_a.detach()
        q_b = q_b.detach()

    lp_b = F.log_softmax(logits_b, dim=1)
    lp_a = F.log_softmax(logits_a, dim=1)

    if symmetric:
        loss = -0.5 * ((q_a * lp_b).sum(dim=1).mean() + (q_b * lp_a).sum(dim=1).mean())
    else:
        loss = -(q_a * lp_b).sum(dim=1).mean()
    return loss


def prototype_orthogonality_loss(protos):
    normalized = F.normalize(protos, p=2, dim=1)
    gram = torch.matmul(normalized, normalized.t())
    identity = torch.eye(gram.size(0), device=protos.device)
    return torch.norm(gram - identity, p='fro') ** 2


def correspondence_loss(z1, z2):
    z1_n = F.normalize(z1, p=2, dim=1)
    z2_n = F.normalize(z2, p=2, dim=1)
    cos = (z1_n * z2_n).sum(dim=1)
    return (1.0 - cos).mean()


def spatial_ranking_loss(z, spatial_coords, k=20, margin=0.5, neg=10):
    device = z.device
    N = z.size(0)

    dist_spat = torch.cdist(spatial_coords, spatial_coords)
    topk_val, topk_idx = torch.topk(dist_spat, k=min(k, N - 1) + 1, dim=1, largest=False)
    neighbors = topk_idx[:, 1:]

    z_norm = F.normalize(z, p=2, dim=1)
    dist_emb = torch.cdist(z_norm, z_norm)
    d_pos = torch.gather(dist_emb, 1, neighbors)

    neg_idx = torch.randint(0, N, size=(N, neg), device=device)
    d_neg = torch.gather(dist_emb, 1, neg_idx)

    loss_per_pair = torch.clamp(d_pos.unsqueeze(2) - d_neg.unsqueeze(1) + margin, min=0.0)
    loss = loss_per_pair.mean()
    return loss


def cellniche_spatial_contrast_loss(z, spatial_coords, k=20, beta=0.0, hard_weight=True, eps=1e-8):
    device = z.device
    N = z.size(0)

    dist_spat = torch.cdist(spatial_coords, spatial_coords)
    topk_val, topk_idx = torch.topk(dist_spat, k=min(k, N - 1) + 1, dim=1, largest=False)
    neighbors = topk_idx[:, 1:]

    if isinstance(beta, float):
        beta = torch.tensor(beta, device=device)
    tau = torch.log1p(torch.exp(beta))

    z_norm = F.normalize(z, p=2, dim=1)
    sim_all = torch.matmul(z_norm, z_norm.t())

    logits = sim_all / tau
    logits = logits - logits.max(dim=1, keepdim=True).values
    self_mask = torch.eye(N, device=device).bool()
    logits_masked = logits.masked_fill(self_mask, -1e9)
    logp = F.log_softmax(logits_masked, dim=1)

    neigh_sim = torch.gather(sim_all, 1, neighbors)
    neigh_logp = torch.gather(logp, 1, neighbors)

    thr = neigh_sim.mean(dim=1, keepdim=True)
    pos = neigh_sim > thr
    neg = ~pos

    if hard_weight:
        w_pos = (1.0 - neigh_sim).clamp(min=0.0)
        w_pos = (w_pos - w_pos.min()) / (w_pos.max() - w_pos.min() + eps)
        w_neg = neigh_sim.clamp(min=0.0)
        w_neg = (w_neg - w_neg.min()) / (w_neg.max() - w_neg.min() + eps) + 1.0
    else:
        w_pos = torch.ones_like(neigh_sim)
        w_neg = torch.ones_like(neigh_sim)

    pos_cnt = pos.sum(dim=1).clamp(min=1).float()
    neg_cnt = neg.sum(dim=1).clamp(min=1).float()
    pull = -(w_pos * pos * neigh_logp).sum(dim=1) / pos_cnt
    push = -(w_neg * neg * neigh_logp).sum(dim=1) / neg_cnt
    loss = (pull + push).mean()
    return loss
