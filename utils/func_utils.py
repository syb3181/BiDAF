import numpy as np


def load_embedding_dict(file_path):
    """
    :param file_path: the file path of embedding dict
    :return: dict[key=word, value=embedding_matrix]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        ret = {}
        for line in f.readlines():
            row = line.split()
            ret[row[0]] = [float(x) for x in row[1:]]
    return ret


def build_embedding_matrix(em_dict: dict, tks:list):
    """
    :param em_dict: embedding_dict
    :param tks: list of vocabulary from text
    :return: FIELD, embedding matrix
    """
    vocab_size = len(tks)
    em_size = len(em_dict[next(iter(em_dict))])
    w = np.zeros((vocab_size, em_size), dtype='float32')
    for i, word in enumerate(tks):
        if word in em_dict:
            w[i] = em_dict[word]
    return w
