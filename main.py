from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import f1_score

import cv2
import os
import numpy as np
import pandas as pd
import time
import math
import argparse
import random
import torch.optim as optim
from tqdm import tqdm

from transformers import AutoModel,AutoTokenizer,ViTModel,ViTFeatureExtractor
from transformers.optimization import get_cosine_schedule_with_warmup

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataloader import CategoryDataset
from dataloader import create_data_loader
from model import TourClassifier


#acc 계산 함수
def calc_tour_acc(pred, label):
    _, idx = pred.max(1)
    acc = torch.eq(idx, label).sum().item() / idx.size()[0]
    x = label.cpu().numpy()
    y = idx.cpu().numpy()
    f1_acc = f1_score(x, y, average='weighted')
    return acc,f1_acc

#평균 및 현재 값을 계산하고 저장
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#시간 재는 함수 분&초
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

#remain 시간을 알려주는 함수 설정(보기편해서 가져옴)
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

#train
def train_epoch(model,data_loader,loss_fn,optimizer,device,scheduler,n_examples,epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    f1_accuracies = AverageMeter()
    sent_count = AverageMeter()

    start = end = time.time()

    model = model.train()
    correct_predictions = 0

    for step,d in enumerate(data_loader):
        data_time.update(time.time() - end)
        #data loader에서의 변수들 불러오기
        batch_size = d["input_ids"].size(0)
        input_ids = d["input_ids"].to(device) #text
        attention_mask = d["attention_mask"].to(device)
        pixel_values = d['pixel_values'].to(device) #image
        cats1 = d["cats1"].to(device)
        cats2 = d["cats2"].to(device)
        cats3 = d["cats3"].to(device)

        outputs,outputs2,outputs3 = model(
          input_ids=input_ids, #text
          attention_mask=attention_mask,
          pixel_values=pixel_values #image
        )
        _, preds = torch.max(outputs3, dim=1)

        loss1 = loss_fn(outputs, cats1)
        loss2 = loss_fn(outputs2, cats2)
        loss3 = loss_fn(outputs3, cats3)

        #loss함수 설정
        loss = loss1 * 0.05 + loss2 * 0.1 + loss3 * 0.85

        correct_predictions += torch.sum(preds == cats3)
        losses.update(loss.item(), batch_size)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)
        if step % 200 == 0 or step == (len(data_loader)-1):
                    acc,f1_acc = calc_tour_acc(outputs3, cats3)
                    accuracies.update(acc, batch_size)
                    f1_accuracies.update(f1_acc, batch_size)


                    print('Epoch: [{0}][{1}/{2}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Elapsed {remain:s} '
                            'Loss: {loss.val:.3f}({loss.avg:.3f}) '
                            'Acc: {acc.val:.3f}({acc.avg:.3f}) '
                            'f1_Acc: {f1_acc.val:.3f}({f1_acc.avg:.3f}) '
                            'sent/s {sent_s:.0f} '
                            .format(
                            epoch, step+1, len(data_loader),
                            data_time=data_time, loss=losses,
                            acc=accuracies,
                            f1_acc=f1_accuracies,
                            remain=timeSince(start, float(step+1)/len(data_loader)),
                            sent_s=sent_count.avg/batch_time.avg
                            )
                        )
    return correct_predictions.double() / n_examples, losses.avg

def validate(model,data_loader,loss_fn,optimizer,device,scheduler,n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    cnt = 0
    for d in tqdm(data_loader):
        with torch.no_grad():
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            pixel_values = d['pixel_values'].to(device)
            cats1 = d["cats1"].to(device)
            cats2 = d["cats2"].to(device)
            cats3 = d["cats3"].to(device)
            outputs,outputs2,outputs3 = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            _, preds = torch.max(outputs3, dim=1)
            loss1 = loss_fn(outputs, cats1)
            loss2 = loss_fn(outputs2, cats2)
            loss3 = loss_fn(outputs3, cats3)

            loss = loss1 * 0.05 + loss2 * 0.1 + loss3 * 0.85

            correct_predictions += torch.sum(preds == cats3)
            losses.append(loss.item())
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if cnt == 0:
                cnt +=1
                outputs3_arr = outputs3
                cats3_arr = cats3
            else:
                outputs3_arr = torch.cat([outputs3_arr, outputs3],0)
                cats3_arr = torch.cat([cats3_arr, cats3],0)
    acc,f1_acc = calc_tour_acc(outputs3_arr, cats3_arr)
    return f1_acc, np.mean(losses)


if __name__ == '__main__':

    df = pd.read_csv('/data2/journey2022/train.csv')
    #target
    le_1 = preprocessing.LabelEncoder()
    le_1.fit(df['cat3'].values)
    df['cat3'] = le_1.transform(df['cat3'].values)

    le_2 = preprocessing.LabelEncoder()
    le_2.fit(df['cat2'].values)
    df['cat2'] = le_2.transform(df['cat2'].values)

    le_3 = preprocessing.LabelEncoder()
    le_3.fit(df['cat1'].values)
    df['cat1'] = le_3.transform(df['cat1'].values)

    folds = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    df['kfold'] = -1 

    for i in range(5):
        df_idx, valid_idx = list(folds.split(df.values, df['cat3']))[i]
        valid = df.iloc[valid_idx]

        df.loc[df[df.id.isin(valid.id) == True].index.to_list(), 'kfold'] = i

    #k-fold를 4개로 나누어 각 k로의 index추가한 데이터 프레임 생성
    df.to_csv('/data2/journey2022/train_folds.csv',index=False)

    device = torch.device("cuda")
    df = pd.read_csv('/data2/journey2022/train_folds.csv')

    train = df[df["kfold"] != 0].reset_index(drop=True)
    valid = df[df["kfold"] == 0].reset_index(drop=True)

    #모델 setting
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch32-384')

    #dataloader setting
    train_data_loader = create_data_loader(train, tokenizer, feature_extractor, 256, 64, shuffle_=True)
    valid_data_loader = create_data_loader(valid, tokenizer, feature_extractor, 256, 64)


    EPOCHS = 15

    #model, optimizer, scheduler, loss functiong setting
    model = TourClassifier(n_classes1 = 6, n_classes2 = 18, n_classes3 = 128, text_model_name = "klue/roberta-large",image_model_name = "google/vit-large-patch32-384").to(device)
    optimizer = optim.AdamW(model.parameters(), lr= 3e-5)
    loss_fn = nn.CrossEntropyLoss().to(device)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps*0.1),
    num_training_steps=total_steps
    )

    max_acc = 0
    for epoch in range(EPOCHS):
        print('-' * 10)
        print(f'Epoch {epoch}/{EPOCHS-1}')
        print('-' * 10)
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train),
            epoch
        )
        validate_acc, validate_loss = validate(
            model,
            valid_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(valid)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')
        print(f'Validate loss {validate_loss} accuracy {validate_acc}')
        print("")
        print("")