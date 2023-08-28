import json
import copy
import yaml
import pickle

import sng_parser


def main():
    split = 'train'  # 'test', 'val_seen', 'val_unseen'
    source = '../tasks/FGR2R/data/FGR2R_{}.json'.format(split)
    target = '../tasks/OBJ_FGR2R/data/OBJ_FGR2R_{}.json'.format(split)

    all_objects = dict()
    all_objects_target = '../tasks/OBJ_FGR2R/data/all_objects_{}.pkl'.format(split)

    with open(source, 'r') as f_:
        data = json.load(f_)

    new_data = copy.deepcopy(data)

    total_length = len(data)

    for idx, item in enumerate(data):
        # for subins objects parsing
        instr_list = yaml.safe_load(item['new_instructions'])
        obj_list = []  # contains objects in all 3 instructions
        obj_origin_list = []   # original words in the sentence
        for instr in instr_list:
            ins_obj = []
            ins_obj_ori = []
            for sub_instr in instr:
                sub_instr_sentence = ' '.join([word for word in sub_instr])
                # print(sub_instr_sentence)
                graph = sng_parser.parse(sub_instr_sentence)

                subins_obj = []
                subins_obj_ori = []

                for entity in graph['entities']:
                    obj_word = str(entity['lemma_head'])
                    # obj_word = str(entity['head'])
                    obj_word_ori = str(entity['span'])

                    subins_obj.append(obj_word)
                    subins_obj_ori.append(obj_word_ori)

                    # add to dictionary of all object words
                    all_objects[obj_word] = 0

                ins_obj.append(subins_obj)
                ins_obj_ori.append(subins_obj_ori)

            obj_list.append(ins_obj)
            obj_origin_list.append(ins_obj_ori)

        # merge into the data dictionary
        new_data[idx]['subins_obj_lemma_list'] = str(obj_list)
        new_data[idx]['subins_obj_ori_list'] = str(obj_origin_list)


        # for instruction objects parsing
        instr_list = item['instructions']
        obj_list = []
        obj_origin_list = []
        for instr in instr_list:
            graph = sng_parser.parse(instr)

            ins_obj = []
            ins_obj_ori = []
            for entity in graph['entities']:
                obj_word = str(entity['lemma_head'])
                # obj_word = str(entity['head'])
                obj_word_ori = str(entity['span'])

                ins_obj.append(obj_word)
                ins_obj_ori.append(obj_word_ori)

            obj_list.append(ins_obj)
            obj_origin_list.append(ins_obj_ori)

        new_data[idx]['ins_obj_lemma_list'] = str(obj_list)
        new_data[idx]['ins_obj_ori_list'] = str(obj_origin_list)

        if idx > 0 and idx % 200 == 0:
            print('{}/{} finished.'.format(idx, total_length))

    with open(target, 'a') as file_:
        json.dump(new_data, file_, ensure_ascii=False, indent=4)
    #
    # with open(all_objects_target, 'wb') as f:
    #     pickle.dump(all_objects, f)


if __name__ == '__main__':
    main()
