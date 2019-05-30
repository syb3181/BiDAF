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

    def __init__(self, data_path, params):
        """
        Loads dataset_params, vocabulary and tags. Ensure you have run `build_vocab.py` on data_dir before using this
        class.
        Args:
            data_dir: (string) directory containing the dataset
            params: (Params) hyperparameters of the training process. This function modifies params and appends
                    dataset_params (such as vocab size, num_of_tags etc.) to params.
        """
        if type(params) == str:
            params = Params(params)
        self.params = params
        self.__build_text_field()
        self.examples = self.load_data(data_path)
        self.dataset = {}
        self.dataset['all'] = Dataset(
            examples=self.examples,
            fields=[v for k, v in self.example_data_fields.items()]
        )
        self.dataset['train'], self.dataset['val'] = self.dataset['all'].split(
            split_ratio=params.split_ratio,
            random_state=random.getstate()
        )
        # updata params
        params.train_size = len(self.dataset['train'].examples)
        params.val_size = len(self.dataset['val'].examples)
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
    def find_ind_in_tk_list(tk_list, target: str):
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
        paragraphs = article['paragraphs']
        for i, paragraph in enumerate(paragraphs):
            context = paragraph['context']
            qas = paragraph['qas']
            for qa in qas:
                query = qa['question']
                for answer in qa['answers']:
                    tc = DataLoader.word_level_tokenize(context)
                    s_ind, t_ind = DataLoader.find_ind_in_tk_list(tc, answer['text'])
                    if s_ind < 0:
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
        return ret

    def load_data(self, data_path):
        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            a = json.load(f)
            articles = a['data']
            for article in articles:
                examples += self.__article_to_examples(article)
        return examples

    def data_iterator(self, split, batch_size):
        return iter(BucketIterator(self.dataset[split], batch_size=batch_size))

    def translate(self, batch, p1_pred, p2_pred):
        """
        :param batch:
        :param preds: list of (start, end)
        :return:
        """
        c, c_lens = batch.c
        ret = []
        pred1 = np.argmax(p1_pred, axis=1)
        pred2 = np.argmax(p2_pred, axis=1)
        preds = [(x, y) for x, y in zip(pred1, pred2)]
        for text, len, (s, t) in zip(c, c_lens, preds):
            tks = DataLoader.word_level_tokenize(text)[:len]
            ret.append(''.join(tks[s: t+1]))
        for x, y in zip(batch.a, ret):
            print(x)
            print(y)
        return ret


if __name__ == '__main__':
    data_loader = DataLoader('../data/dev/dev-v1.1.json', '../data/dataset_configs.json')
    it = data_loader.data_iterator(split='val', batch_size=4)
    a = next(it)
    print(a.c)
    print(a.c.size())
    print(a.c_char.size())
    z = a.c_char.view((4, -1, 20))
    print(z.size())
    for i in range(4):
        tk_list = a.c[i]
        ans_ind = a.ans_ind[i]
        print(ans_ind)
        s_ind, t_ind = ans_ind.numpy()
        print(s_ind, t_ind)
        print([data_loader.WORD_TEXT_FIELD.vocab.itos[ind] for ind in a.a[i]])
        print([data_loader.WORD_TEXT_FIELD.vocab.itos[ind] for ind in a.c[i][s_ind: t_ind + 1]])

