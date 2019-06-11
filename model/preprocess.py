"""
This script is used for the generation of torchtext friendly dataset.
"""

import argparse
import json
import spacy
import nltk
import os
import tqdm
from utils.model_utils import Params

parser = argparse.ArgumentParser(description='Generate dataset!')
parser.add_argument('--data_dir', default='../tmp_data')
ns = parser.parse_args()
params = Params(os.path.join(ns.data_dir, 'dataset_configs.json'))

nlp = spacy.blank('en')


def word_level_tokenize(text):
    return sum(list(map(nltk.word_tokenize, nltk.sent_tokenize(text))), [])


tk_list_path = os.path.join(ns.data_dir, 'words.txt')
with open(tk_list_path, 'r', encoding='utf-8') as f:
    token_set = set([line.strip() for line in f.readlines()])


def spacy_word_level_tokenize(text):
    doc = nlp(text)
    return [token.text for token in doc if token.text != ' ']


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
        token = token.replace("''", '"').replace("``", '"')
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
        self.multi_answer_question = 0

    def process_file(self, file_path):
        examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            a = json.load(f)
            articles = a['data']
            for article in tqdm.tqdm(articles):
                    examples += self.__article_to_examples(article)
            return examples

    def __article_to_examples(self, article: dict):
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
                if len(qa['answers']) > 1:
                    self.multi_answer_question += 1
                q1s, q2s, gts = [], [], []
                for answer in qa['answers']:
                    answer_start = answer['answer_start']
                    answer_end = answer_start + len(answer['text'])
                    q1, q2 = find_index_range(context_spans, answer_start, answer_end)
                    if q1 < 0:
                        raise RuntimeError('Answer not found!')
                    gts.append(answer['text'])
                    q1s.append(q1)
                    q2s.append(q2)
                data = {
                    'c': context,
                    'q': query,
                    'a': gts[-1],
                    'c_word': ' '.join(word_context),
                    'q_word': ' '.join(word_query),
                    'c_char': ' '.join(char_context),
                    'q_char': ' '.join(char_query),
                    'q1': q1s[-1],
                    'q2': q2s[-1],
                    'gts': gts,
                    'q1s': q1s,
                    'q2s': q2s,
                    'tkd_c': word_context
                }
                ret.append(data)
        return ret


if __name__ == '__main__':
    processor = SquadPreprocessor()
    train_data = processor.process_file('../raw_data/train-v1.1.json')
    assert processor.multi_answer_question == 0, "Train data question has multiple answer!"
    val_data = processor.process_file('../raw_data/dev-v1.1.json')
    with open(os.path.join(ns.data_dir, 'train', 'train_data.json'), 'w', encoding='utf-8') as f_train:
        json.dump(train_data, f_train)
    with open(os.path.join(ns.data_dir, 'val', 'val_data.json'), 'w', encoding='utf-8') as f_val:
        json.dump(val_data, f_val)

