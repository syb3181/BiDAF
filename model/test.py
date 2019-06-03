from torchtext.data import Field
from torchtext.data import Example
from torchtext.data import Dataset

# a = 'I am happy .'
#
#
# WORD_TEXT_FIELD = Field(
#     sequential=True,
#     use_vocab=True,
#     batch_first=True,
#     include_lengths=True
# )
#
#
# data_dict = {
#     'text': a
# }
#
# fields_dict = {
#     'text': ('a', WORD_TEXT_FIELD)
# }
#
# example = Example.fromdict(data_dict, fields_dict)
# examples = [x for x in [example] * 5]
# dataset = Dataset(examples=examples, fields=fields_dict.values())
#
# WORD_TEXT_FIELD.build_vocab(dataset)
#
# print(WORD_TEXT_FIELD.vocab.itos)

a = ['1']
import json

with open('../output/test.json', 'w', encoding='utf-8') as f:
    json.dump(a, f)

with open('../output/test.json', 'r', encoding='utf-8') as f:
    a = json.load(f)
    print(type(a))

