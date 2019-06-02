import json
import nltk
import random

import numpy as np

from torchtext.data import Field
from torchtext.data import Dataset
from torchtext.data import Example
from torchtext.data import BucketIterator

from utils.model_utils import Params

random.seed(1110)


class DataLoader(object):
    """
    Handles all aspects of the data.
    """

    def __init__(self, params):
        """
        Loads dataset_params, vocabulary and tags.
        Args:
            params: (Params) hyperparameters of the training process.
            This function modifies params and appends
                    dataset params (such as vocab size, num_of_tags etc.) to params.
        """
        self.params = params
        self.__build_text_field()
        self.dataset = {}
        params.word_vocab_size = len(self.WORD_TEXT_FIELD.vocab.itos)
        params.char_vocab_size = len(self.CHAR_TEXT_FIELD.vocab.itos)
        params.word_vocab = self.WORD_TEXT_FIELD.vocab.itos

    @staticmethod
    def word_level_tokenize(text):
        return sum(list(map(nltk.word_tokenize, nltk.sent_tokenize(text))), [])

    def char_level_tokenize(self, text):
        return sum([
            [y for y in x[:self.params.max_word_len]] + ['<PAD>'] * (self.params.max_word_len - len(x))
            for x in DataLoader.word_level_tokenize(text)
        ], [])

    @staticmethod
    def __find_ind_in_tk_list(tk_list, target: str):
        target = target.replace(' ', '')
        for i in range(len(tk_list)):
            s = tk_list[i]
            j = i + 1
            while j < len(tk_list) and target.startswith(s + tk_list[j]):
                s = s + tk_list[j]
                j += 1
            if target == s:
                return i, j - 1
        return -1, -1

    def __build_text_field(self):
        self.WORD_TEXT_FIELD = Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            tokenize=self.word_level_tokenize,
            include_lengths=True
        )
        self.CHAR_TEXT_FIELD = Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            tokenize=self.char_level_tokenize
        )
        self.INDEX_FIELD = Field(sequential=False, use_vocab=False, is_target=True)
        self.example_data_fields = {
            'c': ('c', self.WORD_TEXT_FIELD),
            'q': ('q', self.WORD_TEXT_FIELD),
            'a': ('a', self.WORD_TEXT_FIELD),
            'c_char': ('c_char', self.CHAR_TEXT_FIELD),
            'q_char': ('q_char', self.CHAR_TEXT_FIELD),
            'ans_ind': ('ans_ind', self.INDEX_FIELD)
        }
        with open(self.params.word_vocab_path, 'r', encoding='utf-8') as f:
            token_list = [word for word in f.read().splitlines()]
            self.WORD_TEXT_FIELD.build_vocab([token_list])
        with open(self.params.char_vocab_path, 'r', encoding='utf-8') as f:
            token_list = [word for word in f.read().splitlines()]
            self.CHAR_TEXT_FIELD.build_vocab([token_list])

    def __article_to_examples(self, article: dict):
        ret = []
        answer_not_found_in_context, answer_tot = 0, 0
        paragraphs = article['paragraphs']
        for i, paragraph in enumerate(paragraphs):
            context = paragraph['context']
            qas = paragraph['qas']
            for qa in qas:
                query = qa['question']
                # normalize quote
                context = context.replace("''", '" ').replace("``", '" ')
                query = query.replace("''", '" ').replace("``", '" ')
                for answer in qa['answers']:
                    tc = DataLoader.word_level_tokenize(context)
                    s_ind, t_ind = DataLoader.__find_ind_in_tk_list(tc, answer['text'])
                    answer_tot += 1
                    if s_ind < 0:
                        print(answer)
                        print(tc)
                        answer_not_found_in_context += 1
                        continue
                    data = {
                        'c': context,
                        'q': query,
                        'a': answer['text'],
                        'c_char': context,
                        'q_char': query,
                        'ans_ind': (s_ind, t_ind),
                    }
                    ret.append(Example.fromdict(data, self.example_data_fields))
        print('{}/{} answer not found in context!'.format(answer_not_found_in_context, answer_tot))
        return ret

    def load_data(self, data_path, split='all'):
        examples = self.__load_data(data_path)
        self.dataset[split] = Dataset(
            examples=examples,
            fields=[v for k, v in self.example_data_fields.items()]
        )

    def split_data(self):
        assert 'all' in self.dataset, "Load data to tab all first!"
        self.dataset['train'], self.dataset['val'] = self.dataset['all'].split(
            split_ratio=self.params.split_ratio,
            random_state=random.getstate()
        )

    def get_dataset_size(self, split):
        assert split in self.dataset, "Tab {} is not loaded!".format(split)
        return len(self.dataset[split].examples)

    def __load_data(self, data_path):
        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            a = json.load(f)
            articles = a['data']
            for article in articles:
                examples += self.__article_to_examples(article)
        return examples

    def data_iterator(self, split, batch_size):
        return iter(BucketIterator(self.dataset[split], batch_size=batch_size))


if __name__ == '__main__':
    params = Params('../data/dataset_configs.json')
    data_loader = DataLoader(params)
    data_path = '../data/train/small.json'
    data_loader.load_data(data_path, 'all')
    data_loader.split_data()
    batch_size = 4
    it = data_loader.data_iterator('train', batch_size=batch_size)
    a = next(it)
    c, c_lens = a.c
    c_char = a.c_char
    print(c.size())
    print(c_char.size())
    z = a.c_char.view((batch_size, -1, params.max_word_len))
    print(z.size())
    for i in range(4):
        tk_list = a.c[i]
        ans_ind = a.ans_ind[i]
        print(ans_ind)
        s_ind, t_ind = ans_ind.numpy()
        print(s_ind, t_ind)
        print([data_loader.WORD_TEXT_FIELD.vocab.itos[ind] for ind in a.a[i]])
        print([data_loader.WORD_TEXT_FIELD.vocab.itos[ind] for ind in a.c[i][s_ind: t_ind + 1]])
