import os

import torch
from torchtext.datasets import SogouNews
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

from model import FastText
from trainer import *
from utils import *

max_epoch = 10
lr = 5
lr_decay = 0.1
step_size = 1
batch_size = 64
clip = 0.1
embedding_size = 64
dropout_p = 0.3

train_iter = SogouNews(split='train')
num_class = len(set([label for (label, text) in train_iter]))
print("num_class = ", num_class)

save_dir = './saved_model/sgnews'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_path = os.path.join(save_dir, 'ckpt.pth')

vocab_size = len(vocab)

model = FastText(vocab_size, embedding_size, num_class, dropout_p)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_decay)

train_iter, test_iter = SogouNews()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

best_accu = None

for epoch in range(0, max_epoch):
    print('-' * 10 + 'epoch: {}/{}'.format(epoch+1, max_epoch))
    epoch_start_time = time.time()
    total_acc = train(model, train_dataloader, criterion, optimizer, clip)
    accu_val = evaluate(model, valid_dataloader)
    if best_accu is not None and best_accu > accu_val:
        scheduler.step()
    else:
        best_accu = accu_val
        torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict()}, save_path)
    print('-' * 59)
    print('epoch {:3d} | time: {:5.2f}s | valid accuracy: {:8.3f} '.format(epoch+1, time.time() - epoch_start_time, accu_val))
    print('-' * 59)


accuracy = evaluate(model, test_dataloader)
print("Fianl accuracy: ", accuracy)

load_path = save_dir + '/ckpt.pth'
checkpoint = torch.load(load_path)
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
print('Best validation epoch: ', epoch)
b_accuracy = evaluate(model, test_dataloader)
print("Best validation accuracy: ", b_accuracy)
