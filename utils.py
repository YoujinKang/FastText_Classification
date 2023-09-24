import torch
from torchtext.datasets import SogouNews
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import ngrams_iterator

def yield_tokens(data_iter):
    for _, text in data_iter:
        tokens = tokenizer(text)
        # yield tokens
        yield list(ngrams_iterator(tokens, 2))

tokenizer = get_tokenizer('basic_english')
train_iter = SogouNews(split='train')
print("-"*10 + 'making vocabulary')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
print("vocab_size = ", len(vocab))

text_pipeline = lambda x: vocab(list(ngrams_iterator(tokenizer(x), 2)))
# text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))  # '1', '2', '3', '4' -> [0, 1, 2, 3]
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)  # [475, 21, 30, 5297]
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)  # input의 누적 합계를 반환
    text_list = torch.cat(text_list)  # batch 내의 모든 단어가 일렬로 들어감 -> nn.Embedding 에 들어가기 위해 하나로 합쳐짐

    return label_list, text_list, offsets