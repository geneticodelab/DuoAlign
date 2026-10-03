import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.W = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.W)
        if adj.is_sparse:
            x = torch.spmm(adj, x)
        else:
            x = torch.mm(adj, x)
        return F.relu(x)


class SharedSpatialEncoder(nn.Module):

    def __init__(self, dim, depth=2):
        super(SharedSpatialEncoder, self).__init__()
        self.depth = depth
        self.layers = nn.ModuleList([GCNLayer(dim, dim) for _ in range(depth)])

    def forward(self, h0, adj_spat):
        z = h0
        for layer in self.layers:
            z = layer(z, adj_spat)
        return z


class FeatureGCN(nn.Module):

    def __init__(self, dim, depth=2):
        super(FeatureGCN, self).__init__()
        self.depth = depth
        self.layers = nn.ModuleList([GCNLayer(dim, dim) for _ in range(depth)])

    def forward(self, h0, adj_feat):
        z = h0
        for layer in self.layers:
            z = layer(z, adj_feat)
        return z


class SAF(nn.Module):

    def __init__(self, dim, fixed_eta=None):
        super(SAF, self).__init__()
        self.fixed_eta = fixed_eta
        self.attn = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, 2))
        self.gcn = GCNLayer(dim, dim)

    def forward(self, z_spat, z_feat, adj_spat, adj_feat):
        a = self.attn(torch.cat([z_spat, z_feat], dim=1))
        a = F.softmax(a, dim=1)
        if self.fixed_eta is None:
            eta = a[:, 1].mean()
        else:
            eta = torch.as_tensor(float(self.fixed_eta), dtype=z_spat.dtype,
                                  device=z_spat.device)
        A_f = (1.0 - eta) * adj_spat + eta * adj_feat
        H = a[:, 0:1] * z_spat + a[:, 1:2] * z_feat
        z_mid = self.gcn(H, A_f)
        return z_mid, A_f, eta


class WeightedFusion(nn.Module):

    def __init__(self, dim, use_weighted_fusion=True):
        super(WeightedFusion, self).__init__()
        self.use_weighted_fusion = use_weighted_fusion
        if use_weighted_fusion:
            self.mlp_w = nn.Sequential(nn.Linear(3 * dim, dim), nn.ReLU(), nn.Linear(dim, 3))
        self.mlp = nn.Sequential(nn.Linear(3 * dim, dim), nn.ReLU())

    def forward(self, z_spat, z_mid, z_feat):
        if self.use_weighted_fusion:
            w = self.mlp_w(torch.cat([z_spat, z_mid, z_feat], dim=1))
            w = F.normalize(w, p=2, dim=1)
        else:
            w = torch.full((z_spat.shape[0], 3), 1.0 / 3.0,
                           dtype=z_spat.dtype, device=z_spat.device)
        z_cat = torch.cat([w[:, 0:1] * z_spat, w[:, 1:2] * z_mid, w[:, 2:3] * z_feat], dim=1)
        return self.mlp(z_cat)


class PrototypePool(nn.Module):

    def __init__(self, n_prototypes, dim):
        super(PrototypePool, self).__init__()
        self.protos = Parameter(torch.randn(n_prototypes, dim))
        torch.nn.init.xavier_uniform_(self.protos)

    def forward(self):
        return self.protos


class AttentionLayer(nn.Module):

    def __init__(self, in_feat, out_feat):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        emb = torch.cat((emb1.unsqueeze(1), emb2.unsqueeze(1)), dim=1)
        v = torch.tanh(torch.matmul(emb, self.w_omega))
        vu = torch.matmul(v, self.u_omega)
        alpha = F.softmax(vu.squeeze(-1) + 1e-6, dim=-1)
        emb_combined = torch.matmul(emb.transpose(1, 2), alpha.unsqueeze(-1)).squeeze(-1)
        return emb_combined, alpha


class Decoder(nn.Module):

    def __init__(self, in_features, out_features):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, out_features)
        )

    def forward(self, z):
        return self.net(z)


class ClusteringLayer(nn.Module):

    def __init__(self, n_clusters, n_z, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.centroids = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_uniform_(self.centroids)

    def forward(self, z):
        dist = torch.sum(torch.pow(z.unsqueeze(1) - self.centroids, 2), 2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


class DuoAlign_Overall(nn.Module):

    def __init__(self, dim_in_1, dim_in_2, dim_out=64, n_prototypes=128, n_clusters=10,
                 tau_pcl=1.0, use_saf=True,
                 share_spatial=True,
                 share_feature=False,
                 share_protos=True,
                 use_cross_attn=True,
                 use_weighted_fusion=True,
                 gcn_depth=2,
                 saf_fixed_eta=None):
        super(DuoAlign_Overall, self).__init__()
        self.dim_out = dim_out
        self.tau_pcl = tau_pcl
        self.use_saf = use_saf
        self.share_spatial = share_spatial
        self.share_feature = share_feature
        self.share_protos = share_protos
        self.use_cross_attn = use_cross_attn
        self.use_weighted_fusion = use_weighted_fusion
        self.gcn_depth = gcn_depth
        self.saf_fixed_eta = saf_fixed_eta

        self.proj1 = nn.Sequential(nn.Linear(dim_in_1, dim_out), nn.ReLU())
        self.proj2 = nn.Sequential(nn.Linear(dim_in_2, dim_out), nn.ReLU())

        if share_spatial:
            self.spat_enc = SharedSpatialEncoder(dim_out, depth=gcn_depth)
        else:
            self.spat_enc1 = SharedSpatialEncoder(dim_out, depth=gcn_depth)
            self.spat_enc2 = SharedSpatialEncoder(dim_out, depth=gcn_depth)

        if share_feature:
            self.feat_enc = FeatureGCN(dim_out, depth=gcn_depth)
        else:
            self.feat_enc1 = FeatureGCN(dim_out, depth=gcn_depth)
            self.feat_enc2 = FeatureGCN(dim_out, depth=gcn_depth)

        self.saf1 = SAF(dim_out, fixed_eta=saf_fixed_eta)
        self.saf2 = SAF(dim_out, fixed_eta=saf_fixed_eta)
        self.fuse1 = WeightedFusion(dim_out, use_weighted_fusion=use_weighted_fusion)
        self.fuse2 = WeightedFusion(dim_out, use_weighted_fusion=use_weighted_fusion)

        if share_protos:
            self.protos = PrototypePool(n_prototypes, dim_out)
        else:
            self.protos1 = PrototypePool(n_prototypes, dim_out)
            self.protos2 = PrototypePool(n_prototypes, dim_out)

        self.atten_cross = AttentionLayer(dim_out, dim_out) if use_cross_attn else None

        self.decoder1 = Decoder(dim_out, dim_in_1)
        self.decoder2 = Decoder(dim_out, dim_in_2)

        self.clustering = ClusteringLayer(n_clusters, dim_out)

    def forward(self, feat1, feat2, adj_spat, adj_feat1, adj_feat2):
        h1 = self.proj1(feat1)
        h2 = self.proj2(feat2)

        if self.share_spatial:
            z_spat1 = self.spat_enc(h1, adj_spat)
            z_spat2 = self.spat_enc(h2, adj_spat)
        else:
            z_spat1 = self.spat_enc1(h1, adj_spat)
            z_spat2 = self.spat_enc2(h2, adj_spat)

        if self.share_feature:
            z_feat1 = self.feat_enc(h1, adj_feat1)
            z_feat2 = self.feat_enc(h2, adj_feat2)
        else:
            z_feat1 = self.feat_enc1(h1, adj_feat1)
            z_feat2 = self.feat_enc2(h2, adj_feat2)

        if self.use_saf:
            z_mid1, A_f1, eta1 = self.saf1(z_spat1, z_feat1, adj_spat, adj_feat1)
            z_mid2, A_f2, eta2 = self.saf2(z_spat2, z_feat2, adj_spat, adj_feat2)
        else:
            z_mid1, z_mid2 = z_spat1, z_spat2
            eta1 = eta2 = None

        z_intra1 = self.fuse1(z_spat1, z_mid1, z_feat1)
        z_intra2 = self.fuse2(z_spat2, z_mid2, z_feat2)

        if self.use_cross_attn:
            z_cross, alpha_cross = self.atten_cross(z_intra1, z_intra2)
        else:
            z_cross = 0.5 * (z_intra1 + z_intra2)
            alpha_cross = None

        recon1 = self.decoder1(z_intra1)
        recon2 = self.decoder2(z_intra2)

        q_cross = self.clustering(z_cross)
        q_intra1 = self.clustering(z_intra1)
        q_intra2 = self.clustering(z_intra2)

        if self.share_protos:
            protos, protos1, protos2 = self.protos.protos, None, None
        else:
            protos, protos1, protos2 = None, self.protos1.protos, self.protos2.protos

        results = {
            'z_spat1': z_spat1, 'z_spat2': z_spat2,
            'z_feat1': z_feat1, 'z_feat2': z_feat2,
            'z_mid1': z_mid1, 'z_mid2': z_mid2,
            'z_intra1': z_intra1, 'z_intra2': z_intra2,
            'z_cross': z_cross,
            'alpha_cross': alpha_cross,
            'eta1': eta1, 'eta2': eta2,
            'recon1': recon1, 'recon2': recon2,
            'q_cross': q_cross, 'q_intra1': q_intra1, 'q_intra2': q_intra2,
            'protos': protos, 'protos1': protos1, 'protos2': protos2,
        }
        return results
