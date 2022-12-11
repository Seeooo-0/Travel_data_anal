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


class CategoryDataset(Dataset):
    def __init__(self, text, image_path, cats1, cats2, cats3, tokenizer, feature_extractor, max_len):
        self.text = text
        self.image_path = image_path
        self.cats1 = cats1
        self.cats2 = cats2
        self.cats3 = cats3
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = str(self.text[idx])
        image_path = os.path.join('/data2/journey2022',str(self.image_path[idx])[2:])
        image = cv2.imread(image_path)
        cat = self.cats1[idx]
        cat2 = self.cats2[idx]
        cat3 = self.cats3[idx]
        #text처리
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding = 'max_length',
            truncation = True,
            return_attention_mask=True, #attention mask를 사용할 것임
            return_tensors='pt',
        )
        #image 처리
        image_feature = self.feature_extractor(images=image, return_tensors="pt")

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(), #attention 연산이 수행되어야 할 token과 무시해야 할 token을 구별하는 정보가 담긴 리스트 0과 1로 되어 있다.
            'pixel_values': image_feature['pixel_values'][0],
            'cats1': torch.tensor(cat, dtype=torch.long),
            'cats2': torch.tensor(cat2, dtype=torch.long),
            'cats3': torch.tensor(cat3, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, feature_extractor, max_len, batch_size, shuffle_=False):
    ds = CategoryDataset(
        text=df.overview.to_numpy(),
        image_path = df.img_path.to_numpy(),
        cats1=df.cat1.to_numpy(),
        cats2=df.cat2.to_numpy(),
        cats3=df.cat3.to_numpy(),
        tokenizer=tokenizer,
        feature_extractor = feature_extractor,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle = shuffle_
    )