import json
import torch

from torchtext.data import Field

from utils.model_utils import Params
from utils.func_utils import random_shuffle


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
        self.shuffler = None
        params.word_vocab_size = len(self.WORD_TEXT_FIELD.vocab.itos)
        params.word_vocab = self.WORD_TEXT_FIELD.vocab.itos

    def __build_text_field(self):
        self.WORD_TEXT_FIELD = Field(
            tokenize=(lambda s: s.split('|')),
            sequential=True,
            use_vocab=True,
            batch_first=True,
            lower=True,
            include_lengths=True
        )
        self.CHAR_TEXT_FIELD = Field(
            tokenize=(lambda s: s.split()),
            sequential=True,
            use_vocab=True,
            batch_first=True,
            lower=True
        )
        self.tensor_fields = {
            'c_word':  self.WORD_TEXT_FIELD,
            'q_word':  self.WORD_TEXT_FIELD,
        }
        self.other_fields = ['c', 'q', 'a', 'q1', 'q2', 'gts', 'q1s', 'q2s', 'tkd_c']
        with open(self.params.word_vocab_path, 'r', encoding='utf-8') as f:
            token_list = [word for word in f.read().splitlines()]
            self.WORD_TEXT_FIELD.build_vocab([token_list])

    def load_data(self, data_path, split='all', size_limit=-1):
        with open(data_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
            self.dataset[split] = examples if size_limit == -1 else examples[:size_limit]

    def split_data(self, split_ratio=0.8, shuffle=False):
        assert 'all' in self.dataset, "Load data to tab all first!"
        dataset = self.dataset['all']
        if shuffle:
            random_shuffle(dataset)
        board = int(len(dataset) * split_ratio)
        self.dataset['train'], self.dataset['val'] = dataset[:board], dataset[board:]

    def get_dataset_size(self, split):
        assert split in self.dataset, "Tab {} is not loaded!".format(split)
        return len(self.dataset[split])

    def data_iterator(self, split, batch_size, shuffle=True):
        """
        Returns a generator that yields batches data with labels. Batch size is params.batch_size. Expires after one
        pass over the data.
        Args:
            data: (dict) contains data which has keys 'data', 'labels' and 'size'
            params: (Params) hyperparameters of the training process.
            shuffle: (bool) whether the data should be shuffled
        Yields:
            batch_data: (Variable) dimension batch_size x seq_len with the sentence data
            batch_labels: (Variable) dimension batch_size x seq_len with the corresponding labels
        """
        assert split in self.dataset, "Tab {} do not exist.".format(split)
        data = self.dataset[split]

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(len(data)))
        if shuffle:
            random_shuffle(order)

        # one pass over data
        for i in range((len(data) + 1) // batch_size):
            # fetch sentences and tags
            data_batch = [data[idx] for idx in order[i * batch_size:(i + 1) * batch_size]]
            batch = {}
            # batch tensor fields
            for field, TEXT_FIELD in self.tensor_fields.items():
                field_batch = [TEXT_FIELD.preprocess(data[field]) for data in data_batch]
                field_batch = TEXT_FIELD.pad(field_batch)
                field_batch = TEXT_FIELD.numericalize(field_batch)
                if torch.cuda.is_available():
                    if type(field_batch) == torch.Tensor:
                        field_batch = field_batch.cuda()
                    if type(field_batch) == tuple:
                        field_batch = [x.cuda() for x in field_batch]
                batch[field] = field_batch
            # batch other fields:
            for field in self.other_fields:
                field_batch = [data[field] for data in data_batch]
                batch[field] = field_batch
            yield batch


if __name__ == '__main__':
    params = Params('../tmp_data/dataset_configs.json')
    data_loader = DataLoader(params)
    data_path = '../tmp_data/train/train_data.json'
    data_loader.load_data(data_path, 'all', size_limit=128)
    data_loader.split_data()
    batch_size = 32
    it = data_loader.data_iterator('train', batch_size=batch_size)
    for a in it:
        c, c_lens = a['c_word']
        q, q_lens = a['q_word']
        context, query, ans = a['c'], a['q'], a['a']
        contexts = a['tkd_c']
        for i in range(batch_size):
            print('-' * 100)
            print(i)
            tk_list = c[i]
            s_ind, t_ind = a['q1'][i], a['q2'][i]
            print("Context: {}".format(context[i]))
            print([data_loader.WORD_TEXT_FIELD.vocab.itos[ind] for ind in c[i]])
            print(s_ind, t_ind)
            print("Query: {}".format(query[i]))
            print([data_loader.WORD_TEXT_FIELD.vocab.itos[ind] for ind in q[i]])
            print("Answer: {}".format(ans[i]))
            print([data_loader.WORD_TEXT_FIELD.vocab.itos[ind] for ind in c[i][s_ind: t_ind + 1]])
            print(a['tkd_c'][i][s_ind: t_ind + 1])

