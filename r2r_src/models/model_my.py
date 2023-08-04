
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
from param import args
import pickle


class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                 dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        if bidirectional:
            print("Using Bidir in EncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        input_size = embedding_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        embeds = self.embedding(inputs)  # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:    # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)

        if args.sub_out == "max":
            ctx_max, _ = ctx.max(1)
            decoder_init = nn.Tanh()(self.encoder2decoder(ctx_max))
        elif args.sub_out == "tanh":
            decoder_init = nn.Tanh()(self.encoder2decoder(h_t))
        else:
            assert False

        ctx = self.drop(ctx)
        if args.zero_init:
            return ctx, torch.zeros_like(decoder_init), torch.zeros_like(c_t)
        else:
            return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/antholo`gy/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, query_dim, ctx_dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(query_dim, ctx_dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(query_dim + ctx_dim, query_dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None,
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

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        if not output_prob:
            attn = logit
        if output_tilde:
            h_tilde = torch.cat((weighted_context, h), 1)
            h_tilde = self.tanh(self.linear_out(h_tilde))
            return h_tilde, attn
        else:
            return weighted_context, attn


class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, embedding_size, hidden_size,
                       dropout_ratio, feature_size=2048+4):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Sequential(
            nn.Linear(args.angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.feat_att_layer = SoftDotAttention(hidden_size, feature_size)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.candidate_att_layer = SoftDotAttention(hidden_size, feature_size)

    def forward(self, action, feature, cand_feat,
                h_0, prev_h1, c_0,
                ctx, ctx_mask=None,
                already_dropfeat=False):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        '''
        action_embeds = self.embedding(action)

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        if not already_dropfeat:
            # Dropout the raw feature as a common regularization
            feature[..., :-args.angle_feat_size] = self.drop_env(feature[..., :-args.angle_feat_size])   # Do not drop the last args.angle_feat_size (position feat)

        prev_h1_drop = self.drop(prev_h1)
        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat((action_embeds, attn_feat), 1)  # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))

        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        if not already_dropfeat:
            cand_feat[..., :-args.angle_feat_size] = self.drop_env(cand_feat[..., :-args.angle_feat_size])

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        return h_1, c_1, logit, h_tilde


class FCNet(nn.Module):
    """
    Simple class for multi-layer non-linear fully connect network
    Activate function: ReLU()
    """
    def __init__(self, dims, dropout=0.0, norm=True):
        super(FCNet, self).__init__()
        self.num_layers = len(dims) - 1
        self.drop = dropout
        self.norm = norm
        self.main = nn.Sequential(*self._init_layers(dims))

    def _init_layers(self, dims):
        layers = []
        for i in range(self.num_layers):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            # layers.append(nn.Dropout(self.drop))
            if self.norm:
                layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            else:
                layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        return layers

    def forward(self, x):
        return self.main(x)


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v)  # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits


class MyNewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, hid_dim, glimpses=1, dropout=0.2):
        super(MyNewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, hid_dim], dropout)
        self.q_proj = FCNet([q_dim, hid_dim], dropout)
        self.drop = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(hid_dim, glimpses), dim=None)

    def attention_step(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v_proj = self.v_proj(v)  # [batch, k, vdim]
        q_proj = self.q_proj(q).unsqueeze(1)  # [batch, 1, qdim]
        logits = self.linear(self.drop(v_proj * q_proj))
        logits = nn.functional.softmax(logits, 1)
        return torch.sum(logits * v[:, torch.arange(v.shape[1]), :], dim=1)

    def forward(self, ctx, detect_feats):
        """
        ctx,  # (batch, length, hidden)
        detect_feats,  # [[batch1_ob1, batch1_ob2, ..], ..., [batchL_ob1, batchL_ob2, ...]]
        """

        att_detect_feats = []
        # print('len of detect_feats', len(detect_feats))
        # print('batch size', len(detect_feats[0]))
        # print('viewpoint feat shape', detect_feats[0][0].shape)

        for l, batch in enumerate(detect_feats):
            att_detect_feats_batch = []
            for i, detect_feat in enumerate(batch):
                if detect_feat.shape[0] == 0:  # no detected objects
                    att_detect_feats_batch.append(torch.zeros((1, 256)).cuda())
                else:
                    att_detect_feats_batch.append(self.attention_step(detect_feat.reshape(1, detect_feat.shape[0], detect_feat.shape[1]), ctx[i, l].unsqueeze(0)))
            att_detect_feats.append(torch.cat(att_detect_feats_batch, dim=0).unsqueeze(1))

        return torch.cat(att_detect_feats, dim=1)


class MyNewAttention_vectorized(nn.Module):
    def __init__(self, v_dim, q_dim, hid_dim, glimpses=1, dropout=0.2):
        super(MyNewAttention_vectorized, self).__init__()

        self.v_proj = FCNet([v_dim, hid_dim], dropout)
        self.q_proj = FCNet([q_dim, hid_dim], dropout)
        self.drop = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(hid_dim, glimpses), dim=None)

    def attention_step(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v_proj = self.v_proj(v)  # [batch, k, vdim]
        q_proj = self.q_proj(q).unsqueeze(1)  # [batch, 1, qdim]
        logits = self.linear(self.drop(v_proj * q_proj))
        logits = nn.functional.softmax(logits, 1)
        return torch.sum(logits * v[:, torch.arange(v.shape[1]), :], dim=1)

    def forward(self, ctx, detect_feats):
        """
        ctx,  # (batch, length, hidden)
        detect_feats,  # [(batch_size, k_batch1, 256), ..., (batch_size, k_batchL, 256)]
        """

        att_detect_feats = []
        # print('len of detect_feats', len(detect_feats))
        # print('batch size', len(detect_feats[0]))
        # print('viewpoint feat shape', detect_feats[0][0].shape)

        for l, feat_batch in enumerate(detect_feats):
            att_detect_feats.append(self.attention_step(feat_batch, ctx[:, l]).unsqueeze(1))

        return torch.cat(att_detect_feats, dim=1)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(args.rnn_dim, args.rnn_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.rnn_dim, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()


class SpeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, action_embeds, feature, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            # x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])
            x[..., :2048] = self.drop3(x[..., :2048])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature
        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x


class SpeakerEncoder_scenecls(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

        # for scene classification
        self.cls_net = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, 256),
            # torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 60),
            # nn.BatchNorm1d(60),
            torch.nn.ReLU()
        )

    def forward(self, action_embeds, feature, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            # x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])
            x[..., :2048] = self.drop3(x[..., :2048])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # v1
        # cls_score = self.cls_net(ctx.reshape((action_embeds.shape[0] * action_embeds.shape[1], -1)))

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature
        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        # v2
        cls_score = self.cls_net(ctx.reshape((action_embeds.shape[0] * action_embeds.shape[1], -1)))

        return x, cls_score


class SpeakerEncoder_detect_v2(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size
        self.detect_feature_size = 256

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.drop4 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size + 256, feature_size)
        self.detect_attention_layer = MyNewAttention_vectorized(self.detect_feature_size, self.hidden_size, hid_dim=256)

        # self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
        #                          batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.post_lstm = nn.LSTM(self.hidden_size + 256, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, action_embeds, feature, detect_feats, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature

        # detect_feat = self.drop4()  # Dropout the detection feature

        x_detect = self.detect_attention_layer(
            ctx,  # (batch, length, hidden)
            detect_feats,  # [(batch_size, k_batch1, 256), ..., (batch_size, k_batchL, 256)]
        )

        # x_detect (batch, length, 256)
        ctx = torch.cat((ctx, x_detect), dim=2)

        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(ctx.shape[0] * ctx.shape[1], -1),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)

        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x


class SpeakerEncoder_detect(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional, detect_feature_size=256, att_hid_dim=256):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size
        self.detect_feature_size = detect_feature_size

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.drop4 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)
        self.detect_attention_layer = MyNewAttention_vectorized(self.detect_feature_size, self.hidden_size, hid_dim=att_hid_dim)

        # self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
        #                          batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.post_lstm = nn.LSTM(self.hidden_size + detect_feature_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

    def forward(self, action_embeds, feature, detect_feats, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature

        # detect_feat = self.drop4()  # Dropout the detection feature

        x_detect = self.detect_attention_layer(
            ctx,  # (batch, length, hidden)
            detect_feats,  # [[batch1_ob1, batch1_ob2, ..], ..., [batchL_ob1, batchL_ob2, ...]]
        )

        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)

        # print(x.shape, x_detect.shape)
        x = torch.cat((x, x_detect), dim=2)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x


class SpeakerEncoder_detect_plusfc(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size
        self.detect_feature_size = 256

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        self.drop4 = nn.Dropout(p=args.featdropout)
        self.attention_layer = SoftDotAttention(self.hidden_size, feature_size)
        # self.detect_attention_layer = MyNewAttention_vectorized(self.detect_feature_size, self.hidden_size, hid_dim=256)
        self.detect_attention_layer = MyNewAttention_vectorized_plusfc(self.detect_feature_size, self.hidden_size, hid_dim=256)

        # self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
        #                          batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.post_lstm = nn.LSTM(self.hidden_size + 256, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)


    def forward(self, action_embeds, feature, detect_feats, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-args.angle_feat_size] = self.drop3(feature[..., :-args.angle_feat_size])   # Dropout the image feature

        # detect_feat = self.drop4()  # Dropout the detection feature

        x_detect = self.detect_attention_layer(
            ctx,  # (batch, length, hidden)
            detect_feats,  # [[batch1_ob1, batch1_ob2, ..], ..., [batchL_ob1, batchL_ob2, ...]]
        )

        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
        )
        x = x.view(batch_size, max_length, -1)

        # print(x.shape, x_detect.shape)
        x = torch.cat((x, x_detect), dim=2)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x


class LandmarkEncoder(nn.Module):
    def __init__(self, encoder_feature_size, dim_feedforward, output_hidden_size, num_layers, num_heads, dropout_ratio):
        super().__init__()
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers  # 3
        self.num_heads = num_heads  # 4
        self.encoder_feature_size = encoder_feature_size
        self.output_hidden_size = output_hidden_size

        self.drop3 = nn.Dropout(p=args.featdropout)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=encoder_feature_size * 2,
                                                        nhead=num_heads,
                                                        dim_feedforward=dim_feedforward,
                                                        dropout=dropout_ratio,
                                                        activation='gelu')
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        self.k_linear = nn.Conv2d(in_channels=encoder_feature_size,
                                  out_channels=encoder_feature_size * 2,
                                  kernel_size=(1, 1),
                                  bias=True)

        self.post_lstm = nn.LSTM(input_size=self.encoder_feature_size * 2, hidden_size=output_hidden_size // 2,
                                 num_layers=2, batch_first=True, dropout=dropout_ratio, bidirectional=True)
        self.drop = nn.Dropout(p=dropout_ratio)

        print('Using Landmark Encoder')

    def forward(self, action_embeds, feature, lengths, already_dropfeat=False):
        """
            :param action_embeds: (batch_size, length, 2052). The feature of the view
            :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
            :param lengths: Not used in it
            :return: context with shape (batch_size, length, hidden_size)
        """
        batch_size, length, embed_size = action_embeds.shape

        x = action_embeds
        # if not already_dropfeat:
        #     x[..., :-args.angle_feat_size] = self.drop3(x[..., :-args.angle_feat_size])  # Do not dropout the spatial features

        x_next = torch.roll(x, -1, 1)
        x_next[:, -1, :] = 0  # (batch_size, length, 2052)
        x_next = x_next.reshape(batch_size * length, embed_size)
        x_next = x_next.unsqueeze(1).expand(batch_size * length, 36, embed_size)
        # print(x_next.shape)

        x_aug = torch.cat((feature.reshape(batch_size * length, 36, embed_size), x_next), 2)  # C_i (batch_size, length, 2052 * 2)
        # print(x_aug.shape)  # torch.Size([320, 36, 4104])

        q = self.encoder(x_aug.transpose(1, 0)).transpose(1, 0)
        q_avg = torch.mean(q, axis=1)
        q_avg = q_avg.reshape(batch_size * length, embed_size * 2, 1)

        # feature (batch_size, length, 36, 2052)
        k = self.k_linear(feature.reshape(batch_size * length * 36, embed_size, 1, 1))
        k = k.reshape(batch_size * length, 36, embed_size * 2)

        a = torch.bmm(k, q_avg)
        a = a.squeeze(2)
        # print(a.shape)
        landmark_id = F.gumbel_softmax(a, hard=True) == 1
        # print(landmark_id.shape)
        # print(landmark_id[0, :])

        l = k[landmark_id]
        l = l.reshape(batch_size, length, embed_size * 2)

        x, _ = self.post_lstm(l)
        x = self.drop(x)

        return x


class SpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0):
        embeds = self.embedding(words)
        embeds = self.drop(embeds)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))

        x = self.drop(x)

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, _ = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous(). view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        return logit, h1, c1


if __name__ == '__main__':
    model = LandmarkEncoder(encoder_feature_size=128,
                            dim_feedforward=64,
                            output_hidden_size=64,
                            num_layers=3,
                            num_heads=1,
                            dropout_ratio=0.1)

    # model = SpeakerEncoder(128,
    #                        64,
    #                        0.1,
    #                        True)

    action_embeds = torch.rand(32, 10, 128)  # (batch_size, length, 2052)
    feature = torch.rand(32, 10, 36, 128)   # (batch_size, length, 36, 2052)

    out = model(action_embeds, feature, None)

    print(out.shape)



