import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
import numpy as np

from .model import DuoAlign_Overall
from .preprocess import fix_seed
from .utils import (evaluate_clustering, sinkhorn_knopp, psa_loss,
                    prototype_contrastive_loss, prototype_orthogonality_loss,
                    correspondence_loss, spatial_ranking_loss,
                    cellniche_spatial_contrast_loss)


os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


class Train_DuoAlign:

    def __init__(self, data_dict, args):
        self.random_state = getattr(args, 'random_state', 2026)
        fix_seed(self.random_state)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.use_dense_adj = getattr(args, 'use_dense_adj', True)

        self.feat_1 = torch.FloatTensor(data_dict['adata_omics1'].obsm['feat']).to(self.device)
        self.feat_2 = torch.FloatTensor(data_dict['adata_omics2'].obsm['feat']).to(self.device)
        self.labels = data_dict['adata_omics1'].obs['Group'].values
        self.n_clusters = len(np.unique(self.labels))

        self.spat_1 = torch.as_tensor(
            np.asarray(data_dict['adata_omics1'].obsm['spatial'], dtype=np.float64),
            dtype=torch.float32).to(self.device)

        graphs = data_dict['graphs']

        def _adj(t):
            t = t.to(self.device)
            return t.to_dense() if self.use_dense_adj else t

        self.adj_spat = _adj(graphs['adj_spatial'])
        self.adj_feat1 = _adj(graphs['adj_feature1'])
        self.adj_feat2 = _adj(graphs['adj_feature2'])
        if self.use_dense_adj:
            print(f"Dense adjacency mode (deterministic): {tuple(self.adj_spat.shape)}")

        dim_in_1, dim_in_2 = self.feat_1.shape[1], self.feat_2.shape[1]
        A = self.args
        self.model = DuoAlign_Overall(
            dim_in_1=dim_in_1,
            dim_in_2=dim_in_2,
            dim_out=A.dim_out,
            n_prototypes=A.n_prototypes,
            n_clusters=self.n_clusters,
            tau_pcl=A.tau_pcl,
            use_saf=A.use_saf,
            share_spatial=getattr(A, 'share_spatial', True),
            share_feature=getattr(A, 'share_feature', False),
            share_protos=getattr(A, 'share_protos', True),
            use_cross_attn=getattr(A, 'use_cross_attn', True),
            use_weighted_fusion=getattr(A, 'use_weighted_fusion', True),
            gcn_depth=getattr(A, 'gcn_depth', 2),
            saf_fixed_eta=getattr(A, 'saf_fixed_eta', None),
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=A.lr, weight_decay=A.weight_decay)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"✅ DuoAlign model initialized ({n_params:,} trainable params)"
              f" | ablations: share_spatial={getattr(A, 'share_spatial', True)}"
              f" share_feature={getattr(A, 'share_feature', False)}"
              f" share_protos={getattr(A, 'share_protos', True)}"
              f" cross_attn={getattr(A, 'use_cross_attn', True)}"
              f" w_fusion={getattr(A, 'use_weighted_fusion', True)}"
              f" depth={getattr(A, 'gcn_depth', 2)}"
              f" fixed_eta={getattr(A, 'saf_fixed_eta', None)}")

    def _forward(self):
        return self.model(self.feat_1, self.feat_2, self.adj_spat, self.adj_feat1, self.adj_feat2)

    def _loss_weights(self):
        A = self.args
        w_psa = getattr(A, 'w_psa', None)
        w_recon = getattr(A, 'w_recon', None)
        w_pcl = getattr(A, 'w_pcl', None)
        w_ortho = getattr(A, 'w_ortho', None)
        if w_psa is None:
            w_psa = A.phi
        if w_recon is None:
            w_recon = A.phi
        if w_pcl is None:
            w_pcl = 1.0 - A.phi
        if w_ortho is None:
            w_ortho = A.lambda_ortho
        return w_psa, w_recon, w_pcl, w_ortho

    def _stage1_loss(self, res):
        A = self.args
        use_sinkhorn = getattr(A, 'use_sinkhorn', True)
        symmetric = getattr(A, 'symmetric', True)
        detach_target = getattr(A, 'detach_target', True)

        loss_psa = psa_loss(res['z_spat1'], res['z_feat1']) + psa_loss(res['z_spat2'], res['z_feat2'])
        loss_recon = F.mse_loss(res['recon1'], self.feat_1) + F.mse_loss(res['recon2'], self.feat_2)

        if res.get('protos') is not None:
            protos1 = protos2 = res['protos']
        else:
            protos1, protos2 = res['protos1'], res['protos2']

        loss_pcl1 = prototype_contrastive_loss(
            res['z_spat1'], res['z_feat1'], protos1,
            tau=A.tau_pcl, eps=A.sinkhorn_eps, iters=A.sinkhorn_iters,
            detach_target=detach_target, use_sinkhorn=use_sinkhorn, symmetric=symmetric)
        loss_pcl2 = prototype_contrastive_loss(
            res['z_spat2'], res['z_feat2'], protos2,
            tau=A.tau_pcl, eps=A.sinkhorn_eps, iters=A.sinkhorn_iters,
            detach_target=detach_target, use_sinkhorn=use_sinkhorn, symmetric=symmetric)
        loss_pcl = loss_pcl1 + loss_pcl2
        loss_ortho = prototype_orthogonality_loss(protos1)
        if protos2 is not protos1:
            loss_ortho = loss_ortho + prototype_orthogonality_loss(protos2)

        w_psa, w_recon, w_pcl, w_ortho = self._loss_weights()
        loss = (w_psa * loss_psa + w_recon * loss_recon +
                w_pcl * loss_pcl + w_ortho * loss_ortho)
        return loss, loss_psa, loss_recon, loss_pcl, loss_ortho

    def _stage2_loss(self, res, p_target):
        protos = res['protos'] if res.get('protos') is not None else res.get('protos1')
        q = res['q_cross']
        loss_kl = F.kl_div(q.log(), p_target, reduction='batchmean')
        loss_cross = prototype_contrastive_loss(
            res['z_intra1'], res['z_intra2'], protos,
            tau=self.args.tau_pcl, eps=self.args.sinkhorn_eps, iters=self.args.sinkhorn_iters)
        loss_recon = F.mse_loss(res['recon1'], self.feat_1) + F.mse_loss(res['recon2'], self.feat_2)
        loss_ortho = prototype_orthogonality_loss(protos)

        loss_front = 0.0
        if not self.args.stage2_freeze_front and getattr(self.args, 'lambda_front', 0.0) > 0:
            loss_front = self.args.lambda_front * (
                psa_loss(res['z_spat1'], res['z_feat1']) + psa_loss(res['z_spat2'], res['z_feat2']) +
                prototype_contrastive_loss(res['z_spat1'], res['z_feat1'], protos,
                                           tau=self.args.tau_pcl, eps=self.args.sinkhorn_eps,
                                           iters=self.args.sinkhorn_iters) +
                prototype_contrastive_loss(res['z_spat2'], res['z_feat2'], protos,
                                           tau=self.args.tau_pcl, eps=self.args.sinkhorn_eps,
                                           iters=self.args.sinkhorn_iters))

        loss = (self.args.gamma * loss_kl +
                self.args.lambda_cross * loss_cross +
                self.args.lambda_recon * loss_recon +
                self.args.lambda_ortho * loss_ortho +
                loss_front)

        loss_rank = 0.0
        if getattr(self.args, 'lambda_rank', 0.0) > 0:
            loss_rank = spatial_ranking_loss(
                res['z_cross'], self.spat_1,
                k=self.args.rank_k, margin=self.args.rank_margin, neg=self.args.rank_neg)

        loss_cellniche = 0.0
        if getattr(self.args, 'lambda_cellniche', 0.0) > 0:
            loss_cellniche = cellniche_spatial_contrast_loss(
                res['z_cross'], self.spat_1,
                k=self.args.cellniche_k, beta=0.0, hard_weight=self.args.cellniche_hard)

        loss = (loss +
                self.args.lambda_rank * loss_rank +
                self.args.lambda_cellniche * loss_cellniche)
        return loss, loss_kl, loss_cross, loss_recon, loss_ortho, loss_rank, loss_cellniche

    def target_distribution(self, q):
        weight = (q ** 2) / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def pretrain(self):
        print("🚀 [Stage 1] Pretraining DuoAlign front-ends "
              f"(phi={self.args.phi}, n_prototypes={self.args.n_prototypes})...")
        self.model.train()

        for epoch in range(self.args.pretrain_epochs):
            self.optimizer.zero_grad()
            res = self._forward()
            loss, loss_psa, loss_recon, loss_pcl, loss_ortho = self._stage1_loss(res)
            loss.backward()
            self.optimizer.step()

            if epoch % 50 == 0 or epoch == self.args.pretrain_epochs - 1:
                with torch.no_grad():
                    self.model.eval()
                    res_eval = self._forward()
                    z_cross = res_eval['z_cross'].cpu().numpy()
                    kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
                    y_pred = kmeans.fit_predict(z_cross)
                    m = evaluate_clustering(self.labels, y_pred)
                    eta1 = res_eval['eta1'].item() if res_eval['eta1'] is not None else float('nan')
                    eta2 = res_eval['eta2'].item() if res_eval['eta2'] is not None else float('nan')
                    print(f"EP {epoch:03d} | Loss {loss.item():.4f} | PSA {loss_psa.item():.4f} | "
                          f"Recon {loss_recon.item():.4f} | PCL {loss_pcl.item():.4f} | "
                          f"Ortho {loss_ortho.item():.4f} | eta1 {eta1:.3f} eta2 {eta2:.3f} | "
                          f"ACC {m['ACC']:.4f} ARI {m['ARI']:.4f}")
                    self.model.train()

        print("✅ Stage 1 complete!")
        return self._forward()['z_cross'].detach()

    def finetune(self, init_z):
        print("\n🚀 [Stage 2] Joint fusion fine-tuning "
              f"(freeze_front={self.args.stage2_freeze_front})...")

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=self.random_state)
        y_pred = kmeans.fit_predict(init_z.cpu().numpy())
        self.model.clustering.centroids.data = torch.tensor(kmeans.cluster_centers_).to(self.device)

        if self.args.stage2_freeze_front:
            frozen_prefix = ('proj1', 'proj2', 'spat_enc', 'feat_enc', 'saf', 'fuse')
            for name, p in self.model.named_parameters():
                if name.startswith(frozen_prefix):
                    p.requires_grad = False
            n_frozen = sum(1 for p in self.model.parameters() if not p.requires_grad)
            print(f"Frozen front-end params: {n_frozen}")

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer_ft = optim.Adam(trainable, lr=self.args.lr * 0.1, weight_decay=self.args.weight_decay)

        final_labels = y_pred

        for epoch in range(self.args.finetune_epochs):
            self.model.train()

            if epoch % 10 == 0:
                with torch.no_grad():
                    self.model.eval()
                    res_eval = self._forward()
                    p_target = self.target_distribution(res_eval['q_cross']).data
                self.model.train()

            optimizer_ft.zero_grad()
            res = self._forward()
            (loss, loss_kl, loss_cross, loss_recon, loss_ortho,
             loss_rank, loss_cellniche) = self._stage2_loss(res, p_target)
            loss.backward()
            optimizer_ft.step()

            if epoch % 50 == 0 or epoch == self.args.finetune_epochs - 1:
                with torch.no_grad():
                    self.model.eval()
                    res_eval = self._forward()
                    y_pred = res_eval['q_cross'].cpu().numpy().argmax(1)
                    m = evaluate_clustering(self.labels, y_pred)
                    rank_str = f"{loss_rank.item():.4f}" if not isinstance(loss_rank, float) else "n/a"
                    cn_str = f"{loss_cellniche.item():.4f}" if not isinstance(loss_cellniche, float) else "n/a"
                    print(f"FT {epoch:03d} | Loss {loss.item():.4f} | KL {loss_kl.item():.4f} | "
                          f"Cross {loss_cross.item():.4f} | Recon {loss_recon.item():.4f} | "
                          f"Ortho {loss_ortho.item():.4f} | Rank {rank_str} | CN {cn_str} | "
                          f"ACC {m['ACC']:.4f} ARI {m['ARI']:.4f}")
                    if epoch == self.args.finetune_epochs - 1:
                        final_labels = y_pred

        print("✅ Stage 2 complete!")
        return final_labels


if __name__ == '__main__':
    class Args:
        random_state = 2026
        use_dense_adj = True

        dim_out = 64
        n_prototypes = 128
        tau_pcl = 0.1
        sinkhorn_eps = 0.05
        sinkhorn_iters = 50
        use_saf = True

        lr = 1e-3
        weight_decay = 5e-4
        pretrain_epochs = 200
        phi = 0.5
        lambda_ortho = 0.01
