# -*- coding: UTF-8 -*-
"""SimGCL integrated into ReChorus.

Reference:
    Yu et al. "Are Graph Augmentations Necessary? Simple Graph
    Contrastive Learning for Recommendation", SIGIR 2022.

The implementation follows the paper's key choices: LightGCN propagation,
layer-wise same-hyperoctant uniform noise, and a joint BPR + InfoNCE loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.BaseModel import GeneralModel
from models.general.LightGCN import LightGCNBase


class SimGCL(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'cl_rate', 'temperature', 'eps', 'include_ego']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of user/item embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=3,
                            help='Number of LightGCN propagation layers.')
        parser.add_argument('--cl_rate', type=float, default=0.2,
                            help='Weight lambda of the contrastive objective.')
        parser.add_argument('--temperature', type=float, default=0.2,
                            help='InfoNCE temperature tau.')
        parser.add_argument('--eps', type=float, default=0.1,
                            help='L2 magnitude of the layer-wise random noise.')
        parser.add_argument('--include_ego', type=int, default=0,
                            help='Include layer-0 embeddings in layer pooling (paper setting: 0).')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.cl_rate = args.cl_rate
        self.temperature = args.temperature
        self.eps = args.eps
        self.include_ego = bool(args.include_ego)

        norm_adj = LightGCNBase.build_adjmat(
            corpus.n_users, corpus.n_items, corpus.train_clicked_set
        )
        self.encoder = SimGCLEncoder(
            self.user_num, self.item_num, self.emb_size, norm_adj,
            self.n_layers, self.eps, self.include_ego
        )

    def forward(self, feed_dict):
        self.check_list = []
        users, items = feed_dict['user_id'], feed_dict['item_id']
        user_all, item_all = self.encoder(perturbed=False)
        user_e, item_e = user_all[users], item_all[items]
        prediction = (user_e[:, None, :] * item_e).sum(dim=-1)
        output = {'prediction': prediction.view(feed_dict['batch_size'], -1)}

        if feed_dict['phase'] == 'train':
            # Two independent noisy graph views.  Unique node ids avoid treating
            # repeated users/items in the mini-batch as false negatives.
            user_view1, item_view1 = self.encoder(perturbed=True)
            user_view2, item_view2 = self.encoder(perturbed=True)
            unique_users = torch.unique(users)
            unique_items = torch.unique(feed_dict['pos_item_id'])
            output.update({
                'user_view1': user_view1[unique_users],
                'user_view2': user_view2[unique_users],
                'item_view1': item_view1[unique_items],
                'item_view2': item_view2[unique_items],
            })
        return output

    def loss(self, output):
        rec_loss = super().loss(output)
        user_cl = self.info_nce(output['user_view1'], output['user_view2'])
        item_cl = self.info_nce(output['item_view1'], output['item_view2'])
        return rec_loss + self.cl_rate * (user_cl + item_cl)

    def info_nce(self, view1, view2):
        view1 = F.normalize(view1, dim=-1)
        view2 = F.normalize(view2, dim=-1)
        logits = torch.matmul(view1, view2.transpose(0, 1)) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)

    class Dataset(GeneralModel.Dataset):
        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            # BaseRunner shuffles candidate positions during training; keep the
            # true positive id explicitly for the item-side contrastive loss.
            feed_dict['pos_item_id'] = int(self.data['item_id'][index])
            return feed_dict


class SimGCLEncoder(nn.Module):
    def __init__(self, user_count, item_count, emb_size, norm_adj,
                 n_layers=3, eps=0.1, include_ego=False):
        super().__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.n_layers = n_layers
        self.eps = eps
        self.include_ego = include_ego

        self.user_emb = nn.Parameter(torch.empty(user_count, emb_size))
        self.item_emb = nn.Parameter(torch.empty(item_count, emb_size))
        nn.init.xavier_uniform_(self.user_emb)
        nn.init.xavier_uniform_(self.item_emb)

        coo = norm_adj.tocoo()
        indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
        values = torch.from_numpy(coo.data).float()
        sparse_adj = torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()
        self.register_buffer('sparse_norm_adj', sparse_adj)

    def forward(self, perturbed=False):
        ego = torch.cat([self.user_emb, self.item_emb], dim=0)
        layer_embeddings = [ego] if self.include_ego else []
        for _ in range(self.n_layers):
            ego = torch.sparse.mm(self.sparse_norm_adj, ego)
            if perturbed:
                random_noise = F.normalize(torch.rand_like(ego), dim=-1)
                ego = ego + torch.sign(ego) * random_noise * self.eps
            layer_embeddings.append(ego)

        all_embeddings = torch.stack(layer_embeddings, dim=1).mean(dim=1)
        return (all_embeddings[:self.user_count],
                all_embeddings[self.user_count:])
