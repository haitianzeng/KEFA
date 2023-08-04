import torch
import numpy as np

import os
import utils
import torch.nn.functional as F
import torch.nn as nn
import pickle

import models.model_ori as model

from models.weight_drop import WeightDrop
from models.pano_gcn import pano_att_gcn_v5

from tslearn.metrics import dtw_path_from_metric



class Speaker():
    env_actions = {
        'left': ([0], [-1], [0]),  # left
        'right': ([0], [1], [0]),  # right
        'up': ([0], [0], [1]),  # up
        'down': ([0], [0], [-1]),  # down
        'forward': ([1], [0], [0]),  # forward
        '<end>': ([0], [0], [0]),  # <end>
        '<start>': ([0], [0], [0]),  # <start>
        '<ignore>': ([0], [0], [0])  # <ignore>
    }

    def __init__(self, env, listener, tok, args):
        self.env = env
        self.feature_size = self.env.feature_size
        self.tok = tok
        self.tok.finalize()
        self.listener = listener
        self.args = args

        # Model
        print("VOCAB_SIZE", self.tok.vocab_size())
        self.encoder = SpeakerEncoder(self.feature_size+self.args.angle_feat_size, self.args.rnn_dim, self.args.dropout, bidirectional=self.args.bidir, args=args).cuda()
        self.decoder = SpeakerDecoder(self.tok.vocab_size(), self.args.wemb, self.tok.word_to_index['<PAD>'],
                                      self.args.rnn_dim, self.args.dropout, args).cuda()
        self.encoder_optimizer = self.args.optimizer(self.encoder.parameters(), lr=self.args.lr)
        self.decoder_optimizer = self.args.optimizer(self.decoder.parameters(), lr=self.args.lr)

        # DTW new
        # self.subins_summarizer = nn.LSTM(self.args.rnn_dim, self.args.rnn_dim, batch_first=True).cuda()

        # Evaluation
        self.softmax_loss = torch.nn.CrossEntropyLoss(ignore_index=self.tok.word_to_index['<PAD>'])

        # Will be used in beam search
        self.nonreduced_softmax_loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.tok.word_to_index['<PAD>'],
            size_average=False,
            reduce=False
        )

        self.acc_iter = 0

        # detection feature
        with open('detect_feat_genome_by_view.pkl', 'rb') as f:
            self.detect_feat_genome_by_view = pickle.load(f)

        with open('vg_class_glove_embed.pkl', 'rb') as f:
            self.vg_class_glove_embed = pickle.load(f)

    def train(self, iters):
        for i in range(iters):
            self.env.reset()

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            loss = self.teacher_forcing(train=True, now_iter=self.acc_iter)

            loss.backward()
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

            self.acc_iter += 1

    def get_insts(self, wrapper=(lambda x: x)):
        # Get the caption for all the data
        self.env.reset_epoch(shuffle=True)
        path2inst = {}
        total = self.env.size()
        for _ in wrapper(range(total // self.env.batch_size + 1)):  # Guarantee that all the data are processed
            obs = self.env.reset()
            insts = self.infer_batch()  # Get the insts of the result
            path_ids = [ob['path_id'] for ob in obs]  # Gather the path ids
            for path_id, inst in zip(path_ids, insts):
                if path_id not in path2inst:
                    path2inst[path_id] = self.tok.shrink(inst)  # Shrink the words
        return path2inst

    def valid(self, *aargs, **kwargs):
        """

        :param iters:
        :return: path2inst: path_id --> inst (the number from <bos> to <eos>)
                 loss: The XE loss
                 word_accu: per word accuracy
                 sent_accu: per sent accuracy
        """
        path2inst = self.get_insts(*aargs, **kwargs)

        # Calculate the teacher-forcing metrics
        self.env.reset_epoch(shuffle=True)
        N = 1 if self.args.fast_train else 3     # Set the iter to 1 if the fast_train (o.w. the problem occurs)
        metrics = np.zeros(3)
        for i in range(N):
            self.env.reset()
            metrics += np.array(self.teacher_forcing(train=False))
        metrics /= N

        return (path2inst, *metrics)

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction([name], [0], [0])
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            # state = self.env.env.sims[idx].getState()
            state = self.env.env.sims[idx].getState()[0]
            if traj is not None:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point) // 12   # The point idx started from 0
                trg_level = (trg_point) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                    # print("UP")
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                    # print("DOWN")
                while self.env.env.sims[idx].getState()[0].viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                    # print("RIGHT")
                    # print(self.env.env.sims[idx].getState().viewIndex, trg_point)
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def _teacher_action(self, obs, ended, tracker=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def _candidate_variable(self, obs, actions):
        candidate_feat = np.zeros((len(obs), self.feature_size + self.args.angle_feat_size), dtype=np.float32)
        for i, (ob, act) in enumerate(zip(obs, actions)):
            if act == -1:  # Ignore or Stop --> Just use zero vector as the feature
                pass
            else:
                c = ob['candidate'][act]
                candidate_feat[i, :] = c['feature'] # Image feat
        return torch.from_numpy(candidate_feat).cuda()

    def from_shortest_path(self, viewpoints=None, get_first_feat=False):
        """
        :param viewpoints: [[], [], ....(batch_size)]. Only for dropout viewpoint
        :param get_first_feat: whether output the first feat
        :return:
        """
        obs = self.env._get_obs()
        ended = np.array([False] * len(obs)) # Indices match permuation of the model, not env
        length = np.zeros(len(obs), np.int64)
        img_feats = []
        can_feats = []
        teacher_action_view_ids = []
        detect_feats = []
        detect_labels = []

        first_feat = np.zeros((len(obs), self.feature_size+self.args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            first_feat[i, -self.args.angle_feat_size:] = utils.angle_feature(ob['heading'], ob['elevation'])
        first_feat = torch.from_numpy(first_feat).cuda()
        while not ended.all():
            if viewpoints is not None:
                for i, ob in enumerate(obs):
                    viewpoints[i].append(ob['viewpoint'])
            img_feats.append(self.listener._feature_variable(obs))
            teacher_action = self._teacher_action(obs, ended)
            teacher_action = teacher_action.cpu().numpy()
            for i, act in enumerate(teacher_action):
                if act < 0 or act == len(obs[i]['candidate']):  # Ignore or Stop
                    teacher_action[i] = -1                      # Stop Action


            with torch.no_grad():
                detect_feat_batch = []
                detect_labels_batch = []
                tea_act_batch = torch.zeros(len(obs)).long()
                for i, ob in enumerate(obs):
                    if teacher_action[i] > -1:
                        teacher_action_viewid = ob['candidate'][teacher_action[i]]['pointId']
                    else:
                        # for stop action, use state.viewIndex as the view for extracting detections
                        teacher_action_viewid = ob['viewIndex']

                    tea_act_batch[i] = int(teacher_action_viewid)

                    view_detects = self.detect_feat_genome_by_view['{}_{}'.format(ob['scan'], ob['viewpoint'])][
                        teacher_action_viewid]
                    glove_embeds = [self.vg_class_glove_embed[int(detect[0])] for detect in view_detects]

                    if len(glove_embeds) > 0:
                        detect_feat_batch.append(torch.stack(glove_embeds, 0).contiguous().cuda().mean(dim=0))
                    else:
                        detect_feat_batch.append(torch.zeros(300).cuda())

                    n_labels = 3

                    detect_label = [int(detect[0]) for detect in view_detects]

                    if len(detect_label) > n_labels:
                        detect_label = detect_label[:n_labels]
                    else:
                        while len(detect_label) < n_labels:
                            detect_label.append(-1)

                    detect_labels_batch.append(torch.LongTensor(detect_label))

                teacher_action_view_ids.append(tea_act_batch)
                detect_feats.append(torch.stack(detect_feat_batch, 0).contiguous())
                detect_labels.append(torch.stack(detect_labels_batch, 0).contiguous())

            can_feats.append(self._candidate_variable(obs, teacher_action))
            self.make_equiv_action(teacher_action, obs)
            length += (1 - ended)
            ended[:] = np.logical_or(ended, (teacher_action == -1))
            obs = self.env._get_obs()
        img_feats = torch.stack(img_feats, 1).contiguous()  # batch_size, max_len, 36, 2052
        can_feats = torch.stack(can_feats, 1).contiguous()  # batch_size, max_len, 2052
        detect_feats = torch.stack(detect_feats, 1).contiguous()  # batch_size, max_len, 300
        detect_labels = torch.stack(detect_labels, 1).contiguous()  # batch_size, max_len, n_labels
        teacher_action_view_ids = torch.stack(teacher_action_view_ids, 1).contiguous()  # batch_size, max_len

        if get_first_feat:
            return (img_feats, can_feats, first_feat, teacher_action_view_ids, detect_labels, detect_feats), length
        else:
            return (img_feats, can_feats, teacher_action_view_ids, detect_labels, detect_feats), length

    def gt_words(self, obs):
        """
        See "utils.Tokenizer.encode_sentence(...)" for "instr_encoding" details
        """
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        return torch.from_numpy(seq_tensor).cuda()

    def dtw_align_loss(self, obs, ctx, x_dtw, att_pred, path_lengths):
        """
        Align sampled representations of sub-instructions to visual feature sequence with DTW.
        x_dtw: contextualized representation of words (i.e. representation to be used in attention)
        """

        subins_pos_list_batch = [ob['subins_pos_list'] for ob in obs]

        loss_dtw_align = 0
        word_cnt = 0
        subins_cnt = 0
        loss_contrast = 0

        # h0 = torch.zeros(1, 1, self.args.rnn_dim).cuda()
        # c0 = torch.zeros(1, 1, self.args.rnn_dim).cuda()

        for i, ob in enumerate(obs):
            acc_len = 0
            summarized_subins = []
            att_pred_subins = []
            for j, subins_pos in enumerate(subins_pos_list_batch[i]):
                st, ed = subins_pos
                ed += 1

                # _, (h1, c1) = self.subins_summarizer(x_dtw[i, st:ed, :].unsqueeze(0), (h0, c0))

                # summarized_subins.append(h1.squeeze(0))

                summarized_subins.append(torch.mean(x_dtw[i, st:ed, :], dim=0).unsqueeze(0))
                acc_len += ed - st

            if len(summarized_subins) == 0:
                continue

            summarized_subins = torch.cat(summarized_subins, dim=0)  # to [n_subins, hidden_size]
            # print(summarized_subins.shape)

            # normalizing to unit norm
            summarized_subins_normed = summarized_subins / torch.norm(summarized_subins, dim=1).unsqueeze(1)
            ctx_normed = ctx[i, :, :] / torch.norm(ctx[i, :, :], dim=1).unsqueeze(1)
            ctx_normed = ctx_normed[:path_lengths[i], :]

            # print('aaa', summarized_subins_normed.shape, ctx_normed.shape)
            att_gt = DTW(summarized_subins_normed, ctx_normed, ctx.shape[1])
            att_gt_expanded = []

            for j, subins_pos in enumerate(subins_pos_list_batch[i]):
                st, ed = subins_pos
                ed += 1

                att_pred_st_ed = att_pred[i, st:ed, :]
                l_ = att_pred_st_ed.shape[0]

                att_pred_subins.append(att_pred_st_ed)

                with torch.no_grad():
                    att_gt_expanded.append(att_gt[j].unsqueeze(0).expand(l_, -1))

            att_pred_subins = torch.cat(att_pred_subins, dim=0)  # to [n_subins, len_path]
            with torch.no_grad():
                att_gt_expanded = torch.cat(att_gt_expanded, dim=0)

            eps = 1e-6

            pos = torch.log((att_pred_subins * att_gt_expanded).sum(dim=(1)) + eps)
            neg = torch.log(1.0 - (att_pred_subins * (1 - att_gt_expanded)).sum(dim=(1)) + eps)

            loss_dtw_align += -1 * (pos.sum() + neg.sum())
            # subins_cnt += att_pred_subins.shape[0]
            word_cnt += att_pred_subins.shape[0]

            # summarized_subins_normed [n_subins, hidden_size]
            inner_products = torch.matmul(summarized_subins_normed, ctx_normed.permute(1, 0))
            inner_products = torch.exp(inner_products * (512 ** -0.5))

            # inner_products [n_subins, max_path_len]
            pos_pair = torch.masked_select(inner_products, att_gt[:, :path_lengths[i]].eq(1))
            neg_pair = torch.masked_select(inner_products, att_gt[:, :path_lengths[i]].eq(0))

            if len(pos_pair) > 0 and len(neg_pair) > 0:
                pos_sum = pos_pair.sum()
                neg_sum = neg_pair.sum()
                loss_contrast += (-1 * torch.log((pos_sum + 1e-5) / (pos_sum + neg_sum)))
                subins_cnt += att_gt.shape[0]

        loss_dtw_align /= subins_cnt
        loss_contrast /= subins_cnt

        return loss_dtw_align

    def teacher_forcing(self, train=True, features=None, insts=None, for_listener=False, now_iter=None):
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        # Get Image Input & Encode
        if features is not None:
            # It is used in calulating the speaker score in beam-search
            assert insts is not None
            (img_feats, can_feats), lengths = features
            ctx = self.encoder(can_feats, img_feats, lengths)
            batch_size = len(lengths)
        else:
            obs = self.env._get_obs()
            batch_size = len(obs)
            (img_feats, can_feats, teacher_action_view_ids, detect_labels, detect_feats), lengths = self.from_shortest_path()      # Image Feature (from the shortest path)
            ctx = self.encoder(can_feats, img_feats, teacher_action_view_ids, detect_labels, detect_feats, lengths)
        h_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()
        ctx_mask = utils.length2mask(lengths)

        # Get Language Input
        if insts is None:
            insts = self.gt_words(obs)                                       # Language Feature

        # For attention loss
        if train:
            # Decode
            logits, _, _, attn, x_dtw = self.decoder(insts, ctx, ctx_mask, h_t, c_t, train=True)
            # print('attn shape', attn.shape)  torch.Size([64, 80, 7])
        else:
            # Decode
            logits, _, _ = self.decoder(insts, ctx, ctx_mask, h_t, c_t)

        # Because the softmax_loss only allow dim-1 to be logit,
        # So permute the output (batch_size, length, logit) --> (batch_size, logit, length)
        logits = logits.permute(0, 2, 1).contiguous()
        loss = self.softmax_loss(
            input  = logits[:, :, :-1],         # -1 for aligning
            target = insts[:, 1:]               # "1:" to ignore the word <BOS>
        )

        if train:
            if now_iter is not None and now_iter >= int(self.args.hparams['warmup_iter']):
                loss_att = self.dtw_align_loss(obs, ctx, x_dtw, attn, lengths)
                # print('loss check', loss, loss_att)  # 6.9075, 2.3872
                loss_att *= self.args.hparams['w_loss_att']
                loss += loss_att

        if for_listener:
            return self.nonreduced_softmax_loss(
                input  = logits[:, :, :-1],         # -1 for aligning
                target = insts[:, 1:]               # "1:" to ignore the word <BOS>
            )

        if train:
            return loss
        else:
            # Evaluation
            _, predict = logits.max(dim=1)                                  # BATCH, LENGTH
            gt_mask = (insts != self.tok.word_to_index['<PAD>'])
            correct = (predict[:, :-1] == insts[:, 1:]) * gt_mask[:, 1:]    # Not pad and equal to gt
            correct, gt_mask = correct.type(torch.LongTensor), gt_mask.type(torch.LongTensor)
            word_accu = correct.sum().item() / gt_mask[:, 1:].sum().item()     # Exclude <BOS>
            sent_accu = (correct.sum(dim=1) == gt_mask[:, 1:].sum(dim=1)).sum().item() / batch_size  # Exclude <BOS>
            return loss.item(), word_accu, sent_accu

    def infer_batch(self, sampling=False, train=False, featdropmask=None):
        """

        :param sampling: if not, use argmax. else use softmax_multinomial
        :param train: Whether in the train mode
        :return: if sampling: return insts(np, [batch, max_len]),
                                     log_probs(torch, requires_grad, [batch,max_len])
                                     hiddens(torch, requires_grad, [batch, max_len, dim})
                      And if train: the log_probs and hiddens are detached
                 if not sampling: returns insts(np, [batch, max_len])
        """
        if train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        # Image Input for the Encoder
        obs = self.env._get_obs()
        batch_size = len(obs)
        viewpoints_list = [list() for _ in range(batch_size)]

        # Get feature
        (img_feats, can_feats, teacher_action_view_ids, detect_labels, detect_feats), lengths = self.from_shortest_path(viewpoints=viewpoints_list)      # Image Feature (from the shortest path)

        # This code block is only used for the featdrop.
        if featdropmask is not None:
            img_feats[..., :-self.args.angle_feat_size] *= featdropmask
            can_feats[..., :-self.args.angle_feat_size] *= featdropmask

        # Encoder
        ctx = self.encoder(can_feats, img_feats, teacher_action_view_ids, detect_labels, detect_feats, lengths,
                           already_dropfeat=(featdropmask is not None))
        ctx_mask = utils.length2mask(lengths)

        # Decoder
        words = []
        log_probs = []
        hidden_states = []
        entropies = []
        h_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, self.args.rnn_dim).cuda()
        ended = np.zeros(len(obs), np.bool)
        word = np.ones(len(obs), np.int64) * self.tok.word_to_index['<BOS>']    # First word is <BOS>
        word = torch.from_numpy(word).view(-1, 1).cuda()
        for i in range(self.args.maxDecode):
            # Decode Step
            logits, h_t, c_t = self.decoder(word, ctx, ctx_mask, h_t, c_t)      # Decode, logits: (b, 1, vocab_size)

            # Select the word
            logits = logits.squeeze()                                           # logits: (b, vocab_size)
            logits[:, self.tok.word_to_index['<UNK>']] = -float("inf")          # No <UNK> in infer
            if sampling:
                probs = F.softmax(logits, -1)
                m = torch.distributions.Categorical(probs)
                word = m.sample()
                log_prob = m.log_prob(word)
                if train:
                    log_probs.append(log_prob)
                    hidden_states.append(h_t.squeeze())
                    entropies.append(m.entropy())
                else:
                    log_probs.append(log_prob.detach())
                    hidden_states.append(h_t.squeeze().detach())
                    entropies.append(m.entropy().detach())
            else:
                values, word = logits.max(1)

            # Append the word
            cpu_word = word.cpu().numpy()
            cpu_word[ended] = self.tok.word_to_index['<PAD>']
            words.append(cpu_word)

            # Prepare the shape for next step
            word = word.view(-1, 1)

            # End?
            ended = np.logical_or(ended, cpu_word == self.tok.word_to_index['<EOS>'])
            if ended.all():
                break

        if train and sampling:
            return np.stack(words, 1), torch.stack(log_probs, 1), torch.stack(hidden_states, 1), torch.stack(entropies, 1)
        else:
            return np.stack(words, 1)       # [(b), (b), (b), ...] --> [b, l]

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        print("Load the speaker's state dict from %s" % path)
        states = torch.load(path)
        def recover_state(name, model, optimizer):
            # print(name)
            # print(list(model.state_dict().keys()))
            # for key in list(model.state_dict().keys()):
            #     print(key, model.state_dict()[key].size())
            state = model.state_dict()
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if self.args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['encoder']['epoch'] - 1



class SpeakerEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size, dropout_ratio, bidirectional, args):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.feature_size = feature_size
        self.args = args

        if bidirectional:
            print("BIDIR in speaker encoder!!")

        self.lstm = nn.LSTM(feature_size, self.hidden_size // self.num_directions, self.num_layers,
                            batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop3 = nn.Dropout(p=args.featdropout)
        # self.attention_layer = model.SoftDotAttention(self.hidden_size, feature_size)
        self.attention_layer = pano_att_gcn_v5(self.hidden_size, feature_size)

        self.post_lstm = nn.LSTM(self.hidden_size, self.hidden_size // self.num_directions, self.num_layers,
                                 batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)

        self.knowledge_module = Entity_Knowledge(fact_dropout=self.args.hparams['fact_dropout'],
                                                 top_k_facts=self.args.hparams['top_k_facts'])

    def forward(self, action_embeds, feature, teacher_action_view_ids, detect_labels, detect_feats, lengths, already_dropfeat=False):
        """
        :param action_embeds: (batch_size, length, 2052). The feature of the view
        :param feature: (batch_size, length, 36, 2052). The action taken (with the image feature)
        :param lengths: Not used in it
        :return: context with shape (batch_size, length, hidden_size)
        """
        x = action_embeds
        if not already_dropfeat:
            x[..., :-self.args.angle_feat_size] = self.drop3(x[..., :-self.args.angle_feat_size])            # Do not dropout the spatial features

        # LSTM on the action embed
        ctx, _ = self.lstm(x)
        ctx = self.drop(ctx)

        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        if not already_dropfeat:
            feature[..., :-self.args.angle_feat_size] = self.drop3(feature[..., :-self.args.angle_feat_size])   # Dropout the image feature

        knowledge = self.knowledge_module(detect_labels)

        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_size),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.feature_size),        # (batch, length, # of images, feature_size) --> (batch x length, # of images, feature_size)
            teacher_action_view_ids.reshape(-1),  # (batch, length) -> (batch * length)
            detect_feats.reshape(-1, 300),  # (batch, length, 300) -> (batch * length, 300)
            knowledge.reshape(-1, 100)
            # None
        )
        x = x.view(batch_size, max_length, -1)
        x = self.drop(x)

        # Post LSTM layer
        x, _ = self.post_lstm(x)
        x = self.drop(x)

        return x



class SpeakerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, hidden_size, dropout_ratio, args):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.args = args

        # self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.lstm = WeightDrop(nn.LSTM(embedding_size, hidden_size, batch_first=False), ['weight_hh_l0'], dropout=self.args.hparams['weight_drop'])

        self.drop = nn.Dropout(dropout_ratio)
        self.attention_layer = model.SoftDotAttention(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)
        self.baseline_projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(128, 1)
        )

    def forward(self, words, ctx, ctx_mask, h0, c0, train=False):
        embeds = self.embedding(words)
        embeds = self.drop(embeds)

        embeds = embeds.permute(1, 0, 2)
        x, (h1, c1) = self.lstm(embeds, (h0, c0))
        # print(x.shape)
        x = x.permute(1, 0, 2)

        x = self.drop(x)

        # Keep x here for DTW align loss
        x_dtw = x

        # Get the size
        batchXlength = words.size(0) * words.size(1)
        multiplier = batchXlength // ctx.size(0)         # By using this, it also supports the beam-search

        # Att and Handle with the shape
        # Reshaping x          <the output> --> (b(word)*l(word), r)
        # Expand the ctx from  (b, a, r)    --> (b(word)*l(word), a, r)
        # Expand the ctx_mask  (b, a)       --> (b(word)*l(word), a)
        x, attn = self.attention_layer(
            x.contiguous().view(batchXlength, self.hidden_size),
            ctx.unsqueeze(1).expand(-1, multiplier, -1, -1).contiguous(). view(batchXlength, -1, self.hidden_size),
            mask=ctx_mask.unsqueeze(1).expand(-1, multiplier, -1).contiguous().view(batchXlength, -1)
        )
        x = x.view(words.size(0), words.size(1), self.hidden_size)

        # Output the prediction logit
        x = self.drop(x)
        logit = self.projection(x)

        if train:
            attn = attn.reshape(words.size(0), words.size(1), -1)
            return logit, h1, c1, attn, x_dtw

        return logit, h1, c1


def cosine_dist(a, b):
    '''
    i.e. inverse cosine similarity
    '''
    return 1 - np.dot(a, b)


def DTW(seq_a, seq_b, b_gt_length, band_width=None):
    """
    DTW is used to find the optimal alignment path;
    Returns GT like 001110000 for each seq_a
    """
    dist_func = cosine_dist

    if band_width is None:
        path, dist = dtw_path_from_metric(seq_a.detach().cpu().numpy(),
                                          seq_b.detach().cpu().numpy(),
                                          metric=dist_func)
    else:
        path, dist = dtw_path_from_metric(seq_a.detach().cpu().numpy(),
                                          seq_b.detach().cpu().numpy(),
                                          sakoe_chiba_radius=band_width,
                                          metric=dist_func)

    with torch.no_grad():
        att_gt = torch.zeros((seq_a.shape[0], b_gt_length)).cuda()

        for i in range(len(path)):
            att_gt[path[i][0], path[i][1]] = 1

        # v2 new: allow overlap
        for i in range(seq_a.shape[0]):
            pos = (att_gt[i] == 1).nonzero(as_tuple=True)[0]
            if pos[0] - 1 >= 0:
                att_gt[i, pos[0] - 1] = 1
            if pos[-1] + 1 < seq_b.shape[0]:
                att_gt[i, pos[-1] + 1] = 1

    return att_gt


class Entity_Knowledge(nn.Module):
    def __init__(self, fact_dropout, top_k_facts):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=300, out_channels=100, kernel_size=(1, 1), padding=0, bias=True)

        self.top_k_facts = top_k_facts

        self.fact_embed_drop = nn.Dropout2d(fact_dropout)

        with torch.no_grad():
            with open('./r2r_src/models/knowledge_rel_embed_v3.pkl', 'rb') as f:
                self.knowledge_fact_dict = pickle.load(f)

            with open('vg_class_glove_embed.pkl', 'rb') as f:
                self.vg_class_glove_embed = pickle.load(f)

            self.vg_class_glove_embed[-1] = torch.zeros(300)
            self.knowledge_fact_dict[-1] = torch.zeros(5, 600)

    def forward(self, detect_labels):
        # detect_labels: batch_size, max_len, n_labels

        batch_size, max_len, n_labels = detect_labels.shape
        detect_labels = detect_labels.reshape(-1)

        with torch.no_grad():
            facts = [self.knowledge_fact_dict[int(label.item())] for _, label in enumerate(detect_labels)]
            facts = torch.stack(facts, dim=0).cuda()  # n_entities, top_k_facts, 600 ([rel_entity_embed, rel_embed])
            if self.top_k_facts < facts.shape[1]:
                facts = facts[:, :self.top_k_facts, :]
            n_entities = facts.shape[0]  # n_entities = batch_size * max_len * n_labels
            facts = facts.reshape(n_entities, self.top_k_facts * 2, 300, 1)
            facts = facts.permute(0, 2, 1, 3)

        x = self.conv(facts)  # (n_entities, 100, self.top_k_facts * 2, 1)
        x = x.permute(0, 2, 1, 3)    # (n_entities, self.top_k_facts * 2, 100, 1)
        x = x.reshape(batch_size * max_len, n_labels * self.top_k_facts * 2, 100)
        x = x.mean(1).reshape(batch_size, max_len, 100)

        final_embed = x  # (batch_size, max_len, 300)
        final_embed = self.fact_embed_drop(final_embed)

        return final_embed
