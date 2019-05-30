import argparse
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', default='../data/train/train-v1.1.json')
    parser.add_argument('--new_data_dir', default='../data/train')
    parser.add_argument('--val_data_path', default='../data/dev/dev-v1.1.json')
    parser.add_argument('--mode', default='all')
    ns = parser.parse_args()
    with open(ns.train_data_path, 'r', encoding='utf-8') as fin_train:
        a = json.load(fin_train)
        new_path = os.path.join(ns.new_data_dir, "{}.json".format(ns.mode))
        if ns.mode == 'small':
            a['data'] = a['data'][:5]
        else:
            with open(ns.val_data_path) as fin_val:
                b = json.load(fin_val)
                a['data'] = a['data'] + b['data']
        with open(new_path, 'w', encoding='utf-8') as fout:
            json.dump(a, fout)
