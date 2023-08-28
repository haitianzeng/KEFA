import json
import nltk
import yaml
from nltk.corpus import wordnet
import pickle

'''
Matching object nouns or phrases to Visual Genome class
notes:
(1) synsets, lemmas, hypernyms and hyponyms
(2) multiple matched words; multiple matched classes
(3) mis-spelling
(4) filter list (e.g. 'you', 'exit', 'left' should not be useful nouns)
(5) vg class synsets (e.g. tree, trees)

'''


# wordnet.synset('dog.n.01').lemma_names( )
# ['dog', 'domestic_dog', 'Canis_familiaris']
# wordnet.synsets("bedroom", pos=wordnet.NOUN)


def main():
    # Load Visual Genome detection classes
    vg_class_path = './KB/data/entities.txt'
    vg_class = {}
    with open(vg_class_path) as f:
        lines = f.readlines()
        for idx, line_ in enumerate(lines):
            line = line_.strip('\n')
            if line.find(',') > -1:  # multiple names for a class
                all_names = line.split(',')
                for name in all_names:
                    vg_class[name.replace(' ', '_')] = idx
            else:
                vg_class[line.replace(' ', '_')] = idx

    # print(vg_class['nightstand'])
    # exit(0)

    # for single word matching
    # vg_class_single_word = {}

    # Load objects (from scene graph parser)
    split = 'train'  # 'test', 'val_seen', 'val_unseen'
    source = '../tasks/OBJ_FGR2R/data/OBJ_FGR2R_{}.json'.format(split)
    save_path = '../tasks/OBJ_FGR2R/data/obj_vg_class_{}.pkl'.format(split)

    with open(source, 'r') as f_:
        data = json.load(f_)

    all_obj_cnt = 0
    all_obj = {}
    obj_vg_class = {}

    for idx, item in enumerate(data):
        all_ins_obj_list = yaml.safe_load(item['ins_obj_lemma_list'])  # v2
        # all_ins_obj_list = yaml.safe_load(item['ins_obj_list'])  # v1
        for ins_obj_list in all_ins_obj_list:
            for obj_ in ins_obj_list:
                obj = obj_
                if obj.find(' ') > -1:
                    obj = obj.replace(' ', '_')  # for wordnet format

                # keep a record in all objects counting
                if obj not in all_obj:
                    all_obj[obj] = 0
                    all_obj_cnt += 1

                if obj in vg_class:
                    # direct matching obj in vg class names
                    obj_vg_class[obj] = vg_class[obj]
                else:
                    # matching synsets, hypernyms and hyponyms in vg class names
                    syn_arr = wordnet.synsets(obj, pos=wordnet.NOUN)
                    matched_flag = False
                    for synset in syn_arr:
                        syn_lemma_names = synset.lemma_names()
                        for syn in syn_lemma_names:
                            if syn in vg_class:
                                obj_vg_class[obj] = vg_class[syn]
                                matched_flag = True
                                break

                        if matched_flag:
                            break

                        hypernyms = synset.hypernyms()
                        for hyper in hypernyms:
                            hyper_lemma_names = hyper.lemma_names()
                            for hyper_name in hyper_lemma_names:
                                if hyper_name in vg_class:
                                    obj_vg_class[obj] = vg_class[hyper_name]
                                    matched_flag = True
                                    break
                            if matched_flag:
                                break

                        if matched_flag:
                            break

                        hyponyms = synset.hyponyms()
                        for hypo in hyponyms:
                            hypo_lemma_names = hypo.lemma_names()
                            for hypo_name in hypo_lemma_names:
                                if hypo_name in vg_class:
                                    obj_vg_class[obj] = vg_class[hypo_name]
                                    matched_flag = True
                                    break
                            if matched_flag:
                                break

                        if matched_flag:
                            break

                    # match last word, like xxx table
                    last_word = obj.split('_')[-1]
                    skip = ['room', 'area']
                    skip_flag = False
                    for s in skip:
                        if last_word.find(s) > -1:
                            skip_flag = True
                            break
                    if skip_flag:
                        continue
                    if last_word in vg_class:
                        # print(last_word, obj)
                        obj_vg_class[obj] = vg_class[last_word]

    cnt = 0
    for k, v in obj_vg_class.items():
        cnt += 1

    print('total objects:', all_obj_cnt)
    print('matched objects:', cnt)

    # with open(save_path, 'wb') as f:
    #     pickle.dump(obj_vg_class, f)


if __name__ == '__main__':
    main()








