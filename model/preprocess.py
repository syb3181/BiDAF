"""
This script is used for the generation of torchtext friendly dataset.
"""

import argparse
import json
import nltk
import os
from utils.model_utils import Params

parser = argparse.ArgumentParser(description='Generate dataset!')
parser.add_argument('--data_dir', default='../data')
ns = parser.parse_args()
params = Params(os.path.join(ns.data_dir, 'dataset_configs.json'))


def word_level_tokenize(text):
    return sum(list(map(nltk.word_tokenize, nltk.sent_tokenize(text))), [])


def char_level_tokenize(text):
    return sum([
        [y for y in x[:params.max_word_len]] + ['<PAD>'] * (params.max_word_len - len(x))
        for x in word_level_tokenize(text)], [])


def get_spans(text, tokens):
    """
    :param text: text
    :param tokens: list of tokens
    :return:
    """
    current = 0
    spans = []
    for token in tokens:
        if token == r"``" or token == r"''":
            token = "\""
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            print(text)
            print(tokens)
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def find_index_range(spans, answer_start, answer_end):
    """
    :param spans: [s0, t0), [s1, t1), ..., [sn, tn)
    :param answer_start:
    :param answer_end:
    :return: index range
    """
    def intersect(span, start, end):
        return not(end <= span[0] or start >= span[1])

    si = []
    for i, span in enumerate(spans):
        if intersect(span, answer_start, answer_end):
            si.append(i)
    return si[0], si[-1]


class SquadPreprocessor:

    def __init__(self):
        pass

    def process_file(self, file_path):
        examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            a = json.load(f)
            articles = a['data']
            for article in articles:
                    examples += self.__article_to_examples(article)
            return examples

    @staticmethod
    def __article_to_examples(article: dict):
        ret = []
        paragraphs = article['paragraphs']
        for i, paragraph in enumerate(paragraphs):
            context = paragraph['context'].replace("''", '" ').replace("``", '" ')
            word_context = word_level_tokenize(context)
            if len(word_context) > params.max_context_len:
                continue
            char_context = char_level_tokenize(context)
            context_spans = get_spans(context, word_context)
            qas = paragraph['qas']
            for qa in qas:
                query = qa['question'].replace("''", '" ').replace("``", '" ')
                word_query = word_level_tokenize(query)
                if len(word_query) > params.max_query_len:
                    continue
                char_query = char_level_tokenize(query)
                for answer in qa['answers']:
                    answer_start = answer['answer_start']
                    answer_end = answer_start + len(answer['text'])
                    s_ind, t_ind = find_index_range(context_spans, answer_start, answer_end)
                    if s_ind < 0:
                        raise RuntimeError('Answer not found!')
                    data = {
                        'c': context,
                        'q': query,
                        'a': answer['text'],
                        'c_word': ' '.join(word_context),
                        'q_word': ' '.join(word_query),
                        'c_char': ' '.join(char_context),
                        'q_char': ' '.join(char_query),
                        'q1': s_ind,
                        'q2': t_ind
                    }
                    ret.append(data)
        return ret


if __name__ == '__main__':
    processor = SquadPreprocessor()
    train_data = processor.process_file('../raw_data/train-v1.1.json')
    val_data = processor.process_file('../raw_data/dev-v1.1.json')
    with open(os.path.join(ns.data_dir, 'train', 'train_data.json'), 'w', encoding='utf-8') as f_train:
        json.dump(train_data, f_train)
    with open(os.path.join(ns.data_dir, 'val', 'val_data.json'), 'w', encoding='utf-8') as f_val:
        json.dump(val_data, f_val)

