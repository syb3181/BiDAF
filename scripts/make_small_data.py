import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--big_path', default='../data/train/train-v1.1.json')
    parser.add_argument('--small_path', default='../data/train/small.json')
    ns = parser.parse_args()
    with open(ns.big_path, 'r', encoding='utf-8') as fin:
        a = json.load(fin)
        articles = a['data']
        a['data'] = articles[:50]
        with open(ns.small_path, 'w', encoding='utf-8') as fout:
            json.dump(a, fout)
