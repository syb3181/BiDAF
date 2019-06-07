"""Build vocabularies of words and tags from datasets"""

import json
import argparse
from utils.model_utils import Params


def save_vocab_to_txt_file(vocab, txt_path):
    """Writes one token per line, 0-based line id corresponds to the id of the token.
    Args:
        vocab: (iterable object) yields token
        txt_path: (stirng) path to vocab file
    """
    with open(txt_path, 'w', encoding='utf-8') as f:
        for token in vocab:
            f.write(token + '\n')


def save_dict_to_json(d, json_path):
    """Saves dict to json file
    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def update_vocab(txt_path, vocab):
    """Update word and tag vocabulary from dataset
    Args:
        txt_path: (string) path to file, one sentence per line
        vocab: (dict or Counter) with update method
    Returns:
        dataset_size: (int) number of elements in the dataset
    """
    with open(txt_path) as f:
        for i, line in enumerate(f):
            vocab.update(line.strip().split(' '))

    return i + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dict_path', default='model_dir/glove.6B.300d.txt')
    parser.add_argument('--words_path', default='tmp_data/words.txt')
    parser.add_argument('--chars_path', default='tmp_data/chars.txt')
    parser.add_argument('--config_file_path', default='tmp_data/dataset_configs.json')
    ns = parser.parse_args()
    words = set()
    chars = set()
    with open(ns.embedding_dict_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            a = line.split()
            words.add(a[0])
            for ch in a[0]:
                chars.add(ch)

    save_vocab_to_txt_file(words, ns.words_path)
    save_vocab_to_txt_file(chars, ns.chars_path)
    params = Params(ns.config_file_path)
    params.save(ns.config_file_path)
