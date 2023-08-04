from __future__ import division
import os
import sys
import subprocess
import threading
import json
import numpy as np
import ast
import tempfile

from .get_stanford_models import get_stanford_models

# Assumes spice.jar is in the same directory as spice.py.  Change as needed.
SPICE_JAR = 'spice-1.0.jar'
TEMP_DIR = 'tmp'
CACHE_DIR = 'cache'


# from https://stackoverflow.com/questions/10459493/
def KMPSearch(pat, txt):
    results = []

    M = len(pat)
    N = len(txt)

    # create lps[] that will hold the longest prefix suffix
    # values for pattern
    lps = [0] * M
    j = 0  # index for pat[]

    # Preprocess the pattern (calculate lps[] array)
    computeLPSArray(pat, M, lps)

    i = 0 # index for txt[]
    while i < N:
        if pat[j] == txt[i]:
            i += 1
            j += 1

        if j == M:
            # print("Found pattern at index " + str(i-j))
            results.append(i - j)
            j = lps[j - 1]

        # mismatch after j matches
        elif i < N and pat[j] != txt[i]:
            # Do not match lps[0..lps[j-1]] characters,
            # they will match anyway
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return results


def computeLPSArray(pat, M, lps):
    len = 0  # length of the previous longest prefix suffix

    assert lps[0] == 0  # lps[0] is always 0
    i = 1

    # the loop calculates lps[i] for i = 1 to M-1
    while i < M:
        if pat[i] == pat[len]:
            len += 1
            lps[i] = len
            i += 1
        else:
            # This is tricky. Consider the example.
            # AAACAAAA and i = 7. The idea is similar
            # to search step.
            if len != 0:
                len = lps[len - 1]

                # Also, note that we do not increment i here
            else:
                lps[i] = 0
                i += 1

# a = [2, 3, 5, 2, 5, 6, 7, 2, 5, 6]
# b = [2, 5, 6]
# KMPSearch(b, a)


def setup(tok):
    # vocab = read_vocab('../tasks/R2R/data/train_vocab.txt')
    # tok = Tokenizer(vocab=vocab, encoding_length=80)

    # directional
    directional = dict()  # left 0, right 1, around 2
    for ins in ['turn right', 'turn to the right', 'make a right', 'veer right', 'take a right']:
        directional[ins] = 0
    for ins in ['turn left', 'turn to the left', 'make a left', 'veer left', 'take a left']:
        directional[ins] = 1
    for ins in ['turn around', 'turn 180 degrees', 'make a 180 degree turn', 'veer around']:
        directional[ins] = 2

    all_directional = []
    all_directional_type = []

    for k, v in directional.items():
        all_directional.append([tok.word_to_index[word] for word in tok.split_sentence(k)])
        all_directional_type.append(v)

    return all_directional, all_directional_type


def parse_action(sentence, all_directional, all_directional_type):
    act_positions = []
    act_types = []

    for i, d_phrase in enumerate(all_directional):
        # print(d_phrase, KMPSearch(d_phrase, ref))
        matching_results = KMPSearch(d_phrase, sentence)
        if len(matching_results) > 0:
            act_positions.extend(matching_results)
            act_types.extend([all_directional_type[i] for _ in range(len(matching_results))])

    act_positions = np.asarray(act_positions)
    act_types = np.asarray(act_types)

    argsort_ = np.argsort(act_positions)

    # act_positions = act_positions[argsort_]
    act_types = act_types[argsort_]

    return act_types


def action_metric(cand, ref, all_directional, all_directional_type):
    '''
        cand, ref should be list of word indices
    '''

    cand_action_types = parse_action(cand, all_directional, all_directional_type)
    ref_action_types = parse_action(ref, all_directional, all_directional_type)

    # print('cand_action_types', cand_action_types)
    # print('ref_action_types', ref_action_types)

    lcs = LCS(cand_action_types, ref_action_types)

    if len(cand_action_types) > 0:
        precision = float(lcs) / len(cand_action_types)
    else:
        precision = 0
    if len(ref_action_types) > 0:
        recall = float(lcs) / len(ref_action_types)
    else:
        recall = 1

    if (precision + recall) > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0

    false_positive = int(len(cand_action_types) - lcs)
    true_positive = int(len(cand_action_types))
    false_negative = int(len(ref_action_types) - lcs)

    # print(precision, recall)

    return precision, recall, f1_score, false_negative, false_positive, true_positive


def LCS(seq_a, seq_b):
    """
        return the length of the longest common subsequence of seq_a and seq_b
    """

    dp = np.zeros((len(seq_a) + 1, len(seq_b) + 1))

    for i, x in enumerate(seq_a):
        for j, y in enumerate(seq_b):
            if x == y:
                dp[i + 1, j + 1] = dp[i, j] + 1
            else:
                dp[i + 1, j + 1] = max(dp[i + 1, j], dp[i, j + 1])

    return dp[len(seq_a), len(seq_b)]


def calculate_action_metric(tok, all_directional, all_directional_type, cand, refs):
    '''
        Warp up multiple references
    '''
    # print(cand)
    # print(refs)
    cand_tok = [tok.word_to_index[word] if word in tok.word_to_index else tok.word_to_index['<UNK>'] for word in tok.split_sentence(cand[0])]
    precision = 0
    recall = 0
    f1_score = -1
    false_negative = 0
    false_positive = 0
    true_positive = 0

    for r in refs:
        ref_tok = [tok.word_to_index[word] if word in tok.word_to_index else tok.word_to_index['<UNK>'] for word in tok.split_sentence(r)]

        pr, re, f, fn, fp, tp = action_metric(cand_tok, ref_tok, all_directional, all_directional_type)
        if f > f1_score:
            precision = pr
            recall = re
            f1_score = f
            false_negative = fn
            false_positive = fp
            true_positive = tp

    return precision, recall, f1_score, false_negative, false_positive, true_positive


class Spice_action_v1:
    """
    Main Class to compute the SPICE metric
    """

    def __init__(self, tok):
        get_stanford_models()
        self.tok = tok
        all_directional, all_directional_type = setup(tok)
        self.all_directional = all_directional
        self.all_directional_type = all_directional_type

    def float_convert(self, obj):
        try:
            return float(obj)
        except:
            return np.nan

    def compute_score(self, gts, res):
        assert (sorted(gts.keys()) == sorted(res.keys()))
        imgIds = sorted(gts.keys())

        # Prepare temp input file for the SPICE scorer
        input_data = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) >= 1)

            input_data.append({
                "image_id": id,
                "test": hypo[0],
                "refs": ref
            })

        cwd = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(cwd, TEMP_DIR)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        in_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir,
                                              mode='w+')
        json.dump(input_data, in_file, indent=2)
        in_file.close()

        # Start job
        out_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        out_file.close()
        cache_dir = os.path.join(cwd, CACHE_DIR)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        spice_cmd = ['java', '-jar', '-Xmx8G', SPICE_JAR, in_file.name,
                     '-cache', cache_dir,
                     '-out', out_file.name,
                     '-subset',
                     '-silent'
                     ]
        subprocess.check_call(spice_cmd,
                              cwd=os.path.dirname(os.path.abspath(__file__)))

        # Read and process results
        with open(out_file.name) as data_file:
            results = json.load(data_file)
        os.remove(in_file.name)
        os.remove(out_file.name)

        imgId_to_scores = {}
        spice_scores = []
        for item in results:
            # new
            hypo = res[item['image_id']]
            ref = gts[item['image_id']]
            pr, re, f, fn, fp, tp = calculate_action_metric(self.tok, self.all_directional,
                                                            self.all_directional_type, hypo, ref)

            # TODO: fix
            # item['scores'].append({
            #     "pr": pr,
            #     "re": re,
            #     "f": f,
            #     "fn": fn,
            #     "numImages": 1,
            #     "fp": fp,
            #     "tp": tp
            # })

            item['scores']['All']['fn'] += fn
            item['scores']['All']['fp'] += fp
            item['scores']['All']['tp'] += tp
            if (item['scores']['All']['tp'] + item['scores']['All']['fp']) > 0:
                item['scores']['All']['pr'] = float(item['scores']['All']['tp']) / (item['scores']['All']['tp'] + item['scores']['All']['fp'])
            if (item['scores']['All']['tp'] + item['scores']['All']['fn']) > 0:
                item['scores']['All']['re'] = float(item['scores']['All']['tp']) / (item['scores']['All']['tp'] + item['scores']['All']['fn'])
            if (item['scores']['All']['pr'] + item['scores']['All']['re']) > 0:
                item['scores']['All']['f'] = 2.0 * item['scores']['All']['pr'] * item['scores']['All']['re'] / (item['scores']['All']['pr'] + item['scores']['All']['re'])

            imgId_to_scores[item['image_id']] = item['scores']
            spice_scores.append(self.float_convert(item['scores']['All']['f']))
        average_score = np.mean(np.array(spice_scores))
        scores = []
        for image_id in imgIds:
            # Convert none to NaN before saving scores over subcategories
            score_set = {}
            for category, score_tuple in imgId_to_scores[image_id].items():
                score_set[category] = {k: self.float_convert(v) for k, v in score_tuple.items()}
            scores.append(score_set)
        return average_score, scores

    def method(self):
        return "SPICE_action_v1"


