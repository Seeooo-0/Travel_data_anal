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

class TourClassifier(nn.Module):
    def __init__(self, n_classes1, n_classes2, n_classes3, text_model_name, image_model_name):
        super(TourClassifier, self).__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name).to(device) #huggingface모델을 사용하기 위해 모델과 매치되는 tokenizer을 불러옴
        self.image_model = ViTModel.from_pretrained(image_model_name).to(device)
        self.text_model.gradient_checkpointing_enable()  #GPU에 한 번에 안 올라가는 모델을 학습하기 위한 테크닉
        self.image_model.gradient_checkpointing_enable() 
        self.drop = nn.Dropout(p=0.1)
        def get_cls(target_size):
            return nn.Sequential(
                nn.Linear(self.text_model.config.hidden_size, self.text_model.config.hidden_size),
                nn.LayerNorm(self.text_model.config.hidden_size),
                nn.Dropout(p = 0.1),
                nn.ReLU(),
                nn.Linear(self.text_model.config.hidden_size, target_size),
            )
        self.cls = get_cls(n_classes1)
        self.cls2 = get_cls(n_classes2)
        self.cls3 = get_cls(n_classes3)

    def forward(self, input_ids, attention_mask,pixel_values):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        image_output = self.image_model(pixel_values = pixel_values)
        #last_hidden_state : 마지막 layer의 hidden state이다
        concat_outputs = torch.cat([text_output.last_hidden_state, image_output.last_hidden_state],1)
        #config hidden size 일치해야함
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.text_model.config.hidden_size, nhead=8).to(device)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2).to(device)

        outputs = transformer_encoder(concat_outputs)
        outputs = outputs[:,0]
        output = self.drop(outputs)
        out1 = self.cls(output)
        out2 = self.cls2(output)
        out3 = self.cls3(output)
        return out1, out2, out3