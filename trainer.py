import time
import torch
from torch.nn.utils import clip_grad_norm_

log_interval = 1000

def train(model, dataloader, criterion, optimizer, clip):
    model.train()
    acc, count = 0, 0
    s_time = time.time()
    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)  # |predicted_label| = (batch, num_classes)
        loss = criterion(predicted_label, label)

        loss.backward()
        clip_grad_norm_(model.parameters(), clip, norm_type=2)
        optimizer.step()

        acc += (predicted_label.argmax(1) == label).sum().item()  # 같으면 1 -> 쭉 더함 
        count += label.size(0)  # batch 때문에 size(0)으로 카운트 셈

        if idx % log_interval == 0 and idx > 0:
            elasped = (time.time() - s_time)
            print('accuracy: {}, time: {}[s]'.format(acc/count, int(elasped)))
            s_time = time.time()   
    return acc/count

def evaluate(model, dataloader):
    model.eval()
    v_total_acc, v_total_count = 0, 0

    with torch.no_grad():
        for (v_label, v_text, v_offsets) in dataloader:
            v_predicted_label = model(v_text, v_offsets)
            v_total_acc += (v_predicted_label.argmax(1) == v_label).sum().item()
            v_total_count += v_label.size(0)

    return v_total_acc/v_total_count