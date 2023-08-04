import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_pano_affinity():
    # get the affinity matrix of panorama, where edges between adjacent views are 1

    # low elevation view 0-11
    # middle elevation view 12-23
    # high elevation view 24-35

    # pano_a = np.zeros((36, 36))  # no self-connect
    pano_a = np.eye(36, dtype=float)  # self-connect

    # low elevation
    for view_id in range(0, 12):
        # up
        pano_a[view_id, view_id + 12] = 1

        # left, left-up
        if view_id == 0:
            pano_a[view_id, 11] = 1
            pano_a[view_id, 11 + 12] = 1
        else:
            pano_a[view_id, view_id - 1] = 1
            pano_a[view_id, view_id - 1 + 12] = 1

        # right, right-up
        if view_id == 11:
            pano_a[view_id, 0] = 1
            pano_a[view_id, 0 + 12] = 1
        else:
            pano_a[view_id, view_id + 1] = 1
            pano_a[view_id, view_id + 1 + 12] = 1


    # middle elevation
    for view_id in range(12, 24):
        # up
        pano_a[view_id, view_id + 12] = 1

        # down
        pano_a[view_id, view_id - 12] = 1

        # left, left-up, left-down
        if view_id == 12:
            pano_a[view_id, 23] = 1
            pano_a[view_id, 23 + 12] = 1
            pano_a[view_id, 23 - 12] = 1
        else:
            pano_a[view_id, view_id - 1] = 1
            pano_a[view_id, view_id - 1 + 12] = 1
            pano_a[view_id, view_id - 1 - 12] = 1

        # right, right-up, right-down
        if view_id == 23:
            pano_a[view_id, 12] = 1
            pano_a[view_id, 12 + 12] = 1
            pano_a[view_id, 12 - 12] = 1
        else:
            pano_a[view_id, view_id + 1] = 1
            pano_a[view_id, view_id + 1 + 12] = 1
            pano_a[view_id, view_id + 1 - 12] = 1


    # high elevation
    for view_id in range(24, 36):
        # down
        pano_a[view_id, view_id - 12] = 1

        # left, left-down
        if view_id == 24:
            pano_a[view_id, 35] = 1
            pano_a[view_id, 35 - 12] = 1
        else:
            pano_a[view_id, view_id - 1] = 1
            pano_a[view_id, view_id - 1 - 12] = 1

        # right, right-down
        if view_id == 35:
            pano_a[view_id, 24] = 1
            pano_a[view_id, 24 - 12] = 1
        else:
            pano_a[view_id, view_id + 1] = 1
            pano_a[view_id, view_id + 1 - 12] = 1

    # checking symmetry
    assert np.sum(pano_a - pano_a.T) == 0

    return pano_a


class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, node_emb, edges, edge_weight):
        node_emb = self.weight(node_emb)
        num_nodes = node_emb.shape[0]
        edge_mitrix = torch.sparse.FloatTensor(edges, edge_weight, (num_nodes, num_nodes))
        #edge_mitrix = gcn_norm(edge_mitrix)
        return torch.sparse.mm(edge_mitrix, node_emb)


class pano_att_gcn(nn.Module):
    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(pano_att_gcn, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

        with torch.no_grad():
            self.pano_a = torch.from_numpy(get_pano_affinity()).float().cuda()
            self.pano_a[self.pano_a.eq(1)] = 0.95
            self.pano_a[self.pano_a.eq(0)] = 0.05

    def forward(self, h, context, teacher_action_view_ids, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        logit = attn

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        # gcn new
        batch, seq_len, ctx_dim = context.shape
        with torch.no_grad():
            pano_a_tea_batch = self.pano_a.unsqueeze(0).expand(batch, -1, -1)[torch.arange(batch), teacher_action_view_ids, :]
            pano_a_tea_batch = pano_a_tea_batch.unsqueeze(1)

        attn3_gcn = attn3 * pano_a_tea_batch

        weighted_context = torch.bmm(attn3_gcn, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn


class pano_att_gcn_v2(nn.Module):
    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(pano_att_gcn_v2, self).__init__()
        self.knowledge_dim = 300
        self.query_dim = query_dim
        self.linear_key = nn.Linear(ctx_dim + self.knowledge_dim, query_dim, bias=False)

        self.linear_query = nn.Linear(query_dim, query_dim, bias=False)
        # self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.scale = query_dim ** -0.5
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

        with torch.no_grad():
            self.pano_a = torch.from_numpy(get_pano_affinity()).float().cuda()
            self.pano_a[self.pano_a.eq(1)] = 0.95
            self.pano_a[self.pano_a.eq(0)] = 0.05

    def forward(self, h, context, teacher_action_view_ids, knowledge_vector, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_query(h).unsqueeze(2)  # batch x dim x 1

        context_know = torch.cat((context, knowledge_vector.unsqueeze(1).expand(-1, 36, -1)), dim=2)
        ba, _, ck_dim = context_know.shape
        context_know = self.linear_key(context_know.reshape(-1, ck_dim))
        context_know = context_know.reshape(ba, 36, self.query_dim)

        # Get attention
        attn = torch.bmm(context_know, target).squeeze(2)  # batch x seq_len
        logit = attn

        # new: add scale
        attn *= self.scale

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))

        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        # gcn new
        batch, seq_len, ctx_dim = context.shape
        with torch.no_grad():
            pano_a_tea_batch = self.pano_a.unsqueeze(0).expand(batch, -1, -1)[torch.arange(batch), teacher_action_view_ids, :]
            pano_a_tea_batch = pano_a_tea_batch.unsqueeze(1)

        attn3_gcn = attn3 * pano_a_tea_batch

        weighted_context = torch.bmm(attn3_gcn, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn


class pano_att_gcn_v3(nn.Module):
    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(pano_att_gcn_v3, self).__init__()
        self.knowledge_dim = 300
        self.query_dim = query_dim
        self.linear_key = nn.Linear(ctx_dim + self.knowledge_dim, query_dim, bias=False)

        self.linear_query = nn.Linear(query_dim, query_dim, bias=False)
        # self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)

        self.linear_value = nn.Linear(ctx_dim, 512, bias=False)

        self.sm = nn.Softmax()
        self.scale = query_dim ** -0.5
        self.linear_out = nn.Linear(query_dim + 512, query_dim, bias=False)
        self.tanh = nn.Tanh()

        with torch.no_grad():
            self.pano_a = torch.from_numpy(get_pano_affinity()).float().cuda()
            self.pano_a[self.pano_a.eq(1)] = 0.95
            self.pano_a[self.pano_a.eq(0)] = 0.05

    def forward(self, h, context, teacher_action_view_ids, knowledge_vector, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_query(h).unsqueeze(2)  # batch x dim x 1

        context_know = torch.cat((context, knowledge_vector.unsqueeze(1).expand(-1, 36, -1)), dim=2)
        ba, _, ck_dim = context_know.shape
        context_know = self.linear_key(context_know.reshape(-1, ck_dim))
        context_know = context_know.reshape(ba, 36, self.query_dim)

        # Get attention
        attn = torch.bmm(context_know, target).squeeze(2)  # batch x seq_len
        logit = attn

        # new: add scale
        attn *= self.scale

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))

        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        # gcn new
        batch, seq_len, ctx_dim = context.shape
        with torch.no_grad():
            pano_a_tea_batch = self.pano_a.unsqueeze(0).expand(batch, -1, -1)[torch.arange(batch), teacher_action_view_ids, :]
            pano_a_tea_batch = pano_a_tea_batch.unsqueeze(1)

        attn3_gcn = attn3 * pano_a_tea_batch

        value = self.linear_value(context.reshape((batch * seq_len, ctx_dim)))
        value = value.reshape((batch, seq_len, 512))
        weighted_context = torch.bmm(attn3_gcn, value).squeeze(1)  # batch x dim

        # weighted_context = torch.bmm(attn3_gcn, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn


class pano_att_gcn_v4(nn.Module):
    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(pano_att_gcn_v4, self).__init__()
        self.query_dim = query_dim
        self.linear_key = nn.Linear(ctx_dim + 300, query_dim, bias=False)

        self.linear_query = nn.Linear(query_dim + 100, query_dim, bias=False)
        # self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.scale = query_dim ** -0.5
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

        with torch.no_grad():
            self.pano_a = torch.from_numpy(get_pano_affinity()).float().cuda()
            self.pano_a[self.pano_a.eq(1)] = 0.95
            self.pano_a[self.pano_a.eq(0)] = 0.05

    def forward(self, h, context, teacher_action_view_ids, detect_feats, knowledge_vector, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        h_know = torch.cat((h, knowledge_vector), dim=1)
        target = self.linear_query(h_know).unsqueeze(2)  # batch x dim x 1

        context_know = torch.cat((context, detect_feats.unsqueeze(1).expand(-1, 36, -1)), dim=2)
        ba, _, ck_dim = context_know.shape
        context_know = self.linear_key(context_know.reshape(-1, ck_dim))
        context_know = context_know.reshape(ba, 36, self.query_dim)

        # Get attention
        attn = torch.bmm(context_know, target).squeeze(2)  # batch x seq_len
        logit = attn

        # new: add scale
        attn *= self.scale

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))

        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        # gcn new
        batch, seq_len, ctx_dim = context.shape
        with torch.no_grad():
            pano_a_tea_batch = self.pano_a.unsqueeze(0).expand(batch, -1, -1)[torch.arange(batch), teacher_action_view_ids, :]
            pano_a_tea_batch = pano_a_tea_batch.unsqueeze(1)

        attn3_gcn = attn3 * pano_a_tea_batch

        # attn3_gcn = attn3

        weighted_context = torch.bmm(attn3_gcn, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn



class pano_att_gcn_v5(nn.Module):
    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(pano_att_gcn_v5, self).__init__()
        self.knowledge_dim = 300
        self.query_dim = query_dim
        self.linear_key = nn.Linear(ctx_dim + self.knowledge_dim, query_dim, bias=False)

        self.linear_query = nn.Linear(query_dim, query_dim, bias=False)
        # self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.scale = query_dim ** -0.5
        self.linear_out = nn.Linear(query_dim + ctx_dim + 100, query_dim, bias=False)
        self.tanh = nn.Tanh()

        with torch.no_grad():
            self.pano_a = torch.from_numpy(get_pano_affinity()).float().cuda()
            self.pano_a[self.pano_a.eq(1)] = 0.95
            self.pano_a[self.pano_a.eq(0)] = 0.05

    def forward(self, h, context, teacher_action_view_ids, detect_feats, knowledge_vector, mask=None,
                output_tilde=True, output_prob=True):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_query(h).unsqueeze(2)  # batch x dim x 1

        context_know = torch.cat((context, detect_feats.unsqueeze(1).expand(-1, 36, -1)), dim=2)
        ba, _, ck_dim = context_know.shape
        context_know = self.linear_key(context_know.reshape(-1, ck_dim))
        context_know = context_know.reshape(ba, 36, self.query_dim)

        # Get attention
        attn = torch.bmm(context_know, target).squeeze(2)  # batch x seq_len
        logit = attn

        # new: add scale
        attn *= self.scale

        if mask is not None:
            # -Inf masking prior to the softmax
            attn.masked_fill_(mask, -float('inf'))

        attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        # gcn new
        batch, seq_len, ctx_dim = context.shape
        with torch.no_grad():
            pano_a_tea_batch = self.pano_a.unsqueeze(0).expand(batch, -1, -1)[torch.arange(batch), teacher_action_view_ids, :]
            pano_a_tea_batch = pano_a_tea_batch.unsqueeze(1)

        attn3_gcn = attn3 * pano_a_tea_batch

        weighted_context = torch.bmm(attn3_gcn, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h, knowledge_vector), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn



# class pano_att_gcn_v5(nn.Module):
#     def __init__(self, query_dim, ctx_dim):
#         '''Initialize layer.'''
#         super(pano_att_gcn_v5, self).__init__()
#         self.knowledge_dim = 300
#         self.query_dim = query_dim
#         self.linear_key = nn.Linear(ctx_dim + self.knowledge_dim + 100, query_dim, bias=False)
#
#         self.linear_query = nn.Linear(query_dim, query_dim, bias=False)
#         # self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
#         self.sm = nn.Softmax()
#         self.scale = query_dim ** -0.5
#         self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
#         self.tanh = nn.Tanh()
#
#         with torch.no_grad():
#             self.pano_a = torch.from_numpy(get_pano_affinity()).float().cuda()
#             self.pano_a[self.pano_a.eq(1)] = 0.95
#             self.pano_a[self.pano_a.eq(0)] = 0.05
#
#     def forward(self, h, context, teacher_action_view_ids, detect_feats, knowledge_vector, mask=None,
#                 output_tilde=True, output_prob=True):
#         '''Propagate h through the network.
#
#         h: batch x dim
#         context: batch x seq_len x dim
#         mask: batch x seq_len indices to be masked
#         '''
#         target = self.linear_query(h).unsqueeze(2)  # batch x dim x 1
#
#         context_know = torch.cat((context, detect_feats.unsqueeze(1).expand(-1, 36, -1), knowledge_vector.unsqueeze(1).expand(-1, 36, -1)), dim=2)
#         ba, _, ck_dim = context_know.shape
#         context_know = self.linear_key(context_know.reshape(-1, ck_dim))
#         context_know = context_know.reshape(ba, 36, self.query_dim)
#
#         # Get attention
#         attn = torch.bmm(context_know, target).squeeze(2)  # batch x seq_len
#         logit = attn
#
#         # new: add scale
#         attn *= self.scale
#
#         if mask is not None:
#             # -Inf masking prior to the softmax
#             attn.masked_fill_(mask, -float('inf'))
#
#         attn = self.sm(attn)    # There will be a bug here, but it's actually a problem in torch source code.
#         attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len
#
#         # gcn new
#         batch, seq_len, ctx_dim = context.shape
#         with torch.no_grad():
#             pano_a_tea_batch = self.pano_a.unsqueeze(0).expand(batch, -1, -1)[torch.arange(batch), teacher_action_view_ids, :]
#             pano_a_tea_batch = pano_a_tea_batch.unsqueeze(1)
#
#         attn3_gcn = attn3 * pano_a_tea_batch
#
#         weighted_context = torch.bmm(attn3_gcn, context).squeeze(1)  # batch x dim
#         if not output_prob:
#             attn = logit
#         if output_tilde:
#             h_tilde = torch.cat((weighted_context, h), 1)
#             h_tilde = self.tanh(self.linear_out(h_tilde))
#             return h_tilde, attn
#         else:
#             return weighted_context, attn


if __name__ == '__main__':
    model = pano_att_gcn(query_dim=512, ctx_dim=2048)

    dummy_q = torch.rand(8, 512)
    dummy_ctx = torch.rand(8, 36, 2048)

    output = model(dummy_q, dummy_ctx)
