#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import hashlib
import logging
import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from packaging import version
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, ViTFeatureExtractor, ViTModel

import lpips
import json
from PIL import Image
import requests
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import torchvision.transforms.functional as TF
from torch.nn.functional import cosine_similarity
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage
from skimage.io import imread
import socket
import piq

# subject_names = [
#     "backpack", "backpack_dog", "bear_plushie", "berry_bowl", "can",
#     "candle", "cat", "cat2", "clock", "colorful_sneaker",
#     "dog", "dog2", "dog3", "dog5", "dog6",
#     "dog7", "dog8", "duck_toy", "fancy_boot", "grey_sloth_plushie",
#     "monster_toy", "pink_sunglasses", "poop_emoji", "rc_car", "red_cartoon",
#     "robot_toy", "shiny_sneaker", "teapot", "vase", "wolf_plushie"
# ]
'''
subject_names = ["colorful_sneaker"]
'''
is_autodl = 'autodl' in socket.gethostname()
subject_names = []
test_num = 4

class PromptDatasetCLIP(Dataset):
    def __init__(self, image_dir, json_file, tokenizer, processor, epoch=None):
        with open(json_file, 'r') as json_file:
            metadata_dict  = json.load(json_file)
        
        self.image_dir = image_dir
        self.image_lst = []
        self.prompt_lst = []
        for key, value in metadata_dict.items():
            if epoch is not None:
                data_dir = os.path.join(self.image_dir, value['data_dir'], str(epoch))
            else:
                data_dir = os.path.join(self.image_dir, value['data_dir'])
            image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if (f.endswith(".png") or f.endswith(".jpg"))]
            self.image_lst.extend(image_files[:test_num])
            class_prompts = [value['instance_prompt']] * len(image_files)
            self.prompt_lst.extend(class_prompts[:test_num])
        
        print('data_list', len(self.image_lst), len(self.prompt_lst))
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_lst)

    def __getitem__(self, idx):
        image_path = self.image_lst[idx]
        image = Image.open(image_path)
        prompt = self.prompt_lst[idx]

        extrema = image.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema):
            return None, None
        else:
            prompt_inputs = self.tokenizer([prompt], padding=True, return_tensors="pt")
            image_inputs = self.processor(images=image, return_tensors="pt")

            return image_inputs, prompt_inputs



class PairwiseImageDatasetCLIP(Dataset):
    def __init__(self, subject, data_dir_A, data_dir_B, processor, epoch):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if (f.endswith(".png") or f.endswith(".jpg"))]

        subject = subject + '-'
        self.image_files_B = []
        # Get image files from each subfolder in data A

        for subfolder in os.listdir(data_dir_B):
    
            if subject in subfolder:
                if epoch is not None:
                    data_dir_B = os.path.join(self.data_dir_B, subfolder, str(epoch))
                else:
                    data_dir_B = os.path.join(self.data_dir_B, subfolder)
   
                image_files_b = [os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B) if (f.endswith(".png") or f.endswith(".jpg"))]
                self.image_files_B.extend(image_files_b[:test_num])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = processor

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            inputs_A = self.processor(images=image_A, return_tensors="pt")
            inputs_B = self.processor(images=image_B, return_tensors="pt")

            return inputs_A, inputs_B


class PairwiseImageDatasetDINO(Dataset):
    def __init__(self, subject, data_dir_A, data_dir_B, feature_extractor, epoch):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
 
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if (f.endswith(".png") or f.endswith(".jpg"))]

        subject = subject + '-'
        self.image_files_B = []
	    # Get image files from each subfolder in data A

        for subfolder in os.listdir(data_dir_B):

            if subject in subfolder:
                if epoch is not None:
                    data_dir_B = os.path.join(self.data_dir_B, subfolder, str(epoch))
                else:
                    data_dir_B = os.path.join(self.data_dir_B, subfolder)
                image_files_b = [os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B) if (f.endswith(".png") or f.endswith(".jpg"))]        
                self.image_files_B.extend(image_files_b[:test_num])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            inputs_A = self.feature_extractor(images=image_A, return_tensors="pt")
            inputs_B = self.feature_extractor(images=image_B, return_tensors="pt")

            return inputs_A, inputs_B


class SelfPairwiseImageDatasetCLIP(Dataset):
    def __init__(self, subject, data_dir, processor):
        self.data_dir_A = data_dir
        self.data_dir_B = data_dir
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if (f.endswith(".png") or f.endswith(".jpg"))]

        self.data_dir_B = os.path.join(self.data_dir_B, subject)
        self.image_files_B = [os.path.join(self.data_dir_B, f) for f in os.listdir(self.data_dir_B) if (f.endswith(".png") or f.endswith(".jpg"))]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = processor

    def __len__(self):
        return len(self.image_files_A) * (len(self.image_files_B) - 1)

    def __getitem__(self, index):
        index_A = index // (len(self.image_files_B) - 1)
        index_B = index % (len(self.image_files_B) - 1)

        # Ensure we don't have the same index for A and B
        if index_B >= index_A:
            index_B += 1

        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")
        
        inputs_A = self.processor(images=image_A, return_tensors="pt")
        inputs_B = self.processor(images=image_B, return_tensors="pt")

        return inputs_A, inputs_B


class SelfPairwiseImageDatasetDINO(Dataset):
    def __init__(self, subject, data_dir, feature_extractor):
        self.data_dir_A = data_dir
        self.data_dir_B = data_dir
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if (f.endswith(".png") or f.endswith(".jpg"))]

        self.data_dir_B = os.path.join(self.data_dir_B, subject)
        self.image_files_B = [os.path.join(self.data_dir_B, f) for f in os.listdir(self.data_dir_B) if (f.endswith(".png") or f.endswith(".jpg"))]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_files_A) * (len(self.image_files_B) - 1)

    def __getitem__(self, index):
        index_A = index // (len(self.image_files_B) - 1)
        index_B = index % (len(self.image_files_B) - 1)

        # Ensure we don't have the same index for A and B
        if index_B >= index_A:
            index_B += 1

        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")
        
        inputs_A = self.feature_extractor(images=image_A, return_tensors="pt")
        inputs_B = self.feature_extractor(images=image_B, return_tensors="pt")
        return inputs_A, inputs_B


class PairwiseImageDatasetNOREF(Dataset):
    def __init__(self, subject, data_dir_A, data_dir_B, feature_extractor, epoch, cuda='False'):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
 
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if (f.endswith(".png") or f.endswith(".jpg"))]

        subject = subject + '-'
        self.image_files_B = []
	    # Get image files from each subfolder in data A

        for subfolder in os.listdir(data_dir_B):

            if subject in subfolder:
                if epoch is not None:
                    data_dir_B = os.path.join(self.data_dir_B, subfolder, str(epoch))
                else:
                    data_dir_B = os.path.join(self.data_dir_B, subfolder)
                image_files_b = [os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B) if (f.endswith(".png") or f.endswith(".jpg"))]        
                self.image_files_B.extend(image_files_b[:test_num])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor
        self.cuda = cuda

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)
        
        #image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        #image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")
        image_A = torch.tensor(imread(self.image_files_A[index_A])).permute(2, 0, 1)[None, ...] / 255.
        image_B = torch.tensor(imread(self.image_files_B[index_B])).permute(2, 0, 1)[None, ...] / 255.
        if self.cuda:
            image_A = image_A.cuda()
            image_B = image_B.cuda()
        inputs_A = self.feature_extractor(image_A)
        inputs_B = self.feature_extractor(image_B)

        return inputs_A, inputs_B


class SelfPairwiseImageDatasetLPIPS(Dataset):
    def __init__(self, subject, data_dir):
        self.data_dir_A = data_dir
        self.data_dir_B = data_dir
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if (f.endswith(".png") or f.endswith(".jpg"))]
        self.data_dir_B = os.path.join(self.data_dir_B, subject)
        self.image_files_B = [os.path.join(self.data_dir_B, f) for f in os.listdir(self.data_dir_B) if (f.endswith(".png") or f.endswith(".jpg"))]

        self.transform = Compose([
            Resize((512, 512)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_files_A) * (len(self.image_files_B) - 1)

    def __getitem__(self, index):
        index_A = index // (len(self.image_files_B) - 1)
        index_B = index % (len(self.image_files_B) - 1)

        # Ensure we don't have the same index for A and B
        if index_B >= index_A:
            index_B += 1
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            if self.transform:
                image_A = self.transform(image_A)
                image_B = self.transform(image_B)

            return image_A, image_B


class PairwiseImageDatasetLPIPS(Dataset):
    def __init__(self, subject, data_dir_A, data_dir_B, epoch):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if (f.endswith(".png") or f.endswith(".jpg"))]

        subject = subject + '-'
        self.image_files_B = []
        # Get image files from each subfolder in data A
        for subfolder in os.listdir(data_dir_B):
            if subject in subfolder:
                if epoch is not None:
                    data_dir_B = os.path.join(self.data_dir_B, subfolder, str(epoch))
                else:
                    data_dir_B = os.path.join(self.data_dir_B, subfolder)
                image_files_b = [os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B) if (f.endswith(".png") or f.endswith(".jpg"))]
                self.image_files_B.extend(image_files_b[:test_num])

        self.transform = Compose([
            Resize((512, 512)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            if self.transform:
                image_A = self.transform(image_A)
                image_B = self.transform(image_B)

            return image_A, image_B


def clip_text(image_dir, epoch=None, meta_data_path=None):
    criterion = 'clip_text'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_autodl:
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        # Get the text features
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        # Get the image features
        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    else:
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir='./hub', local_files_only=False).to(device)
        # Get the text features
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14", cache_dir='./hub', local_files_only=False)
        # Get the image features
        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir='./hub', local_files_only=False)

    dataset = PromptDatasetCLIP(image_dir, meta_data_path, tokenizer, processor, epoch)
    dataloader = DataLoader(dataset, batch_size=32)

    similarity = []
    for i in tqdm(range(len(dataset))):
        image_inputs, prompt_inputs = dataset[i]
        if image_inputs is not None and prompt_inputs is not None:
            image_inputs['pixel_values'] = image_inputs['pixel_values'].to(device)
            prompt_inputs['input_ids'] = prompt_inputs['input_ids'].to(device)
            prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'].to(device)
            # print(prompt_inputs)
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**prompt_inputs)

            sim = cosine_similarity(image_features, text_features)

            #image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            #text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            #logit_scale = model.logit_scale.exp()
            #sim = torch.matmul(text_features, image_features.t()) * logit_scale
            similarity.append(sim.item())

    mean_similarity = torch.tensor(similarity).mean().item()
    print(criterion, 'mean_similarity', mean_similarity)

    return mean_similarity, criterion


def clip_image(image_dir, epoch=None):
    criterion = 'clip_image'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_autodl:
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        # Get the image features
        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    else:
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir='./hub', local_files_only=True).to(device)
        # Get the image features
        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir='./hub', local_files_only=True)

    similarity = []
    for subject in subject_names:
        dataset = PairwiseImageDatasetCLIP(subject, './data', image_dir, processor, epoch)
        # dataset = SelfPairwiseImageDatasetCLIP(subject, './data', processor)

        for i in tqdm(range(len(dataset))):
            inputs_A, inputs_B = dataset[i]
            if inputs_A is not None and inputs_B is not None:
                inputs_A['pixel_values'] = inputs_A['pixel_values'].to(device)
                inputs_B['pixel_values'] = inputs_B['pixel_values'].to(device) 

                image_A_features = model.get_image_features(**inputs_A)
                image_B_features = model.get_image_features(**inputs_B)

                image_A_features = image_A_features / image_A_features.norm(p=2, dim=-1, keepdim=True)
                image_B_features = image_B_features / image_B_features.norm(p=2, dim=-1, keepdim=True)
            
                logit_scale = model.logit_scale.exp()
                sim = torch.matmul(image_A_features, image_B_features.t()) # * logit_scale
                similarity.append(sim.item())

    mean_similarity = torch.tensor(similarity).mean().item()
    print(criterion, 'mean_similarity', mean_similarity)

    return mean_similarity, criterion


def dino(image_dir, epoch=None):
    criterion = 'dino'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_autodl:
        model = ViTModel.from_pretrained('facebook/dino-vits16').to(device)
        feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')
    else:
        model = ViTModel.from_pretrained('facebook/dino-vits16', cache_dir='./hub', local_files_only=True).to(device)
        feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16', cache_dir='./hub', local_files_only=True)

    similarity = []

    for subject in subject_names:
        dataset = PairwiseImageDatasetDINO(subject, './data', image_dir, feature_extractor, epoch)
        # dataset = SelfPairwiseImageDatasetDINO(subject, './data', feature_extractor)

        for i in tqdm(range(len(dataset))):
            inputs_A, inputs_B = dataset[i]
            if inputs_A is not None and inputs_B is not None:
                inputs_A['pixel_values'] = inputs_A['pixel_values'].to(device)
                inputs_B['pixel_values'] = inputs_B['pixel_values'].to(device) 

                outputs_A = model(**inputs_A)
                image_A_features = outputs_A.last_hidden_state[:, 0, :]

                outputs_B = model(**inputs_B)
                image_B_features = outputs_B.last_hidden_state[:, 0, :]

                image_A_features = image_A_features / image_A_features.norm(p=2, dim=-1, keepdim=True)
                image_B_features = image_B_features / image_B_features.norm(p=2, dim=-1, keepdim=True)

                sim = torch.matmul(image_A_features, image_B_features.t()) # * logit_scale
                similarity.append(sim.item())

    mean_similarity = torch.tensor(similarity).mean().item()
    print(criterion, 'mean_similarity', mean_similarity)

    return mean_similarity, criterion



class PairwiseImageDatasetLPIPSUPDATED(Dataset):
    def __init__(self, subject, data_dir_A, data_dir_B, epoch, src=False):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
        
        # self.data_dir_A = os.path.join(self.data_dir_A, subject)
        # self.image_files_A = [os.path.join(self.data_dir_A, f) for f in os.listdir(self.data_dir_A) if (f.endswith(".png") or f.endswith(".jpg"))]
        if not src:
            subject = subject + '-'
            self.test_num = test_num
        else:
            self.test_num = len(os.listdir(os.path.join(self.data_dir_B, subject)))

        self.image_files_B = []
        # Get image files from each subfolder in data G
        for subfolder in os.listdir(data_dir_B):
            if (src and subject == subfolder) or (not src) and (subject in subfolder):
                if epoch is not None:
                    data_dir_B = os.path.join(self.data_dir_B, subfolder, str(epoch))
                else:
                    data_dir_B = os.path.join(self.data_dir_B, subfolder)
                image_files_b = [os.path.join(data_dir_B, f) for f in os.listdir(data_dir_B) if (f.endswith(".png") or f.endswith(".jpg"))]
                self.image_files_B.extend(image_files_b[:self.test_num])

        self.image_files_A = []
        # Get image files from each subfolder in data A
        for subfolder in os.listdir(data_dir_A):
            if (src and subject == subfolder) or (not src) and (subject in subfolder):
                if epoch is not None:
                    data_dir_A = os.path.join(self.data_dir_A, subfolder, str(epoch))
                else:
                    data_dir_A = os.path.join(self.data_dir_A, subfolder)
                image_files_b = [os.path.join(data_dir_A, f) for f in os.listdir(data_dir_A) if (f.endswith(".png") or f.endswith(".jpg"))]
                self.image_files_A.extend(image_files_b[:self.test_num])

        self.transform = Compose([
            Resize((512, 512)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_files_A) * self.test_num

    def __getitem__(self, index):
        group_imgs_num = self.test_num * self.test_num
        group_index = index // group_imgs_num
        group_id = index % group_imgs_num
        
        index_A = group_index * self.test_num + group_id//self.test_num
        index_B = group_index * self.test_num + group_id % self.test_num
        # print(index_A, index_B)
        image_A = Image.open(self.image_files_A[index_A]).convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]).convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            if self.transform:
                image_A = self.transform(image_A)
                image_B = self.transform(image_B)

            return image_A, image_B


def lpips_image(image_dir, epoch=None, src=False):
    criterion = 'lpips_image_updated'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up the LPIPS model (vgg=True uses the VGG-based model from the paper)
    # loss_fn = lpips.LPIPS(net='vgg').to(device)
    loss_fn = lpips.LPIPS(net='alex').to(device)
    similarity = []
    for subject in subject_names:
        if src:
            dataset = PairwiseImageDatasetLPIPSUPDATED(subject, './data', './data', epoch, src=True)
        else:
            dataset = PairwiseImageDatasetLPIPSUPDATED(subject, image_dir, image_dir, epoch)

        for i in tqdm(range(len(dataset))):
            image_A, image_B = dataset[i]
            if image_A is not None and image_B is not None:
                image_A = image_A.to(device)
                image_B = image_B.to(device)

                # Calculate LPIPS between the two images
                distance = loss_fn(image_A, image_B)

                similarity.append(distance.item())
    while 0 in similarity:
        similarity.remove(0)
    mean_similarity = torch.tensor(similarity).mean().item()
    print(criterion, 'LPIPS distance', mean_similarity)

    return mean_similarity, criterion


def no_ref_image_BRISQUE(image_dir, epoch=None, criterion='BRISQUE'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if criterion == 'BRISQUE':
        extractor = piq.brisque
        cuda = False
    elif criterion == 'CLIPIQA':
        extractor = piq.CLIPIQA(data_range=1.).to('cuda')
        cuda = True

    else: 
        raise ValueError('No such criterion', criterion)
    score = []
    ori_score = []
    for subject in subject_names:
        dataset = PairwiseImageDatasetNOREF(subject, './data', image_dir, extractor, epoch, cuda=cuda)
        for i in tqdm(range(len(dataset))):
            inputs_A, inputs_B = dataset[i]
            score.append(inputs_B)
            ori_score.append(inputs_A)
    mean_score = torch.tensor(score).mean().item()
    mean_score_ori = torch.tensor(ori_score).mean().item()
    print(criterion, 'mean_score', mean_score)
    print(criterion, 'mean_score_for_ori_data', mean_score_ori)
    return mean_score, criterion


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--subject_name",
        type=str,
        nargs='+',
        default=['backpack'],
        help="List of subject names"
    )
    parser.add_argument(
        "--json_name",
        type=str,
        default='metadata_1.json',
    )
    parser.add_argument(
        "--src",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    image_dirs = []
    args = parse_args()
    epoch = None
    
    subject_names = args.subject_name
    image_dirs.append(args.image_dir)
    function_list = [clip_text, dino, clip_image, lpips_image, no_ref_image_BRISQUE]
    
    for image_dir in image_dirs:
        if epoch:
            name = image_dir + '-' + str(epoch)
        else:
            name = image_dir
        for i, func in enumerate(function_list):
            if i == 0:
                sim, criterion = func(image_dir, epoch, args.json_name)
            elif i == 3:
                sim, criterion = func(image_dir, epoch, args.src)
            elif i == len(function_list)-1:
                sim, criterion = func(image_dir, epoch, 'CLIPIQA')
            else:
                sim, criterion = func(image_dir, epoch)
            if len(subject_names) >= 10:
                filename = f"{args.image_dir}/{len(subject_names)}_classes"  # the name of the file to save the value to
            
            else:
                filename = f"{args.image_dir}/results_updated_{args.subject_name}"  # the name of the file to save the value to
            if args.src:
                filename = f"{filename}_src.txt"
            else:
                filename = f"{filename}.txt"
            # Check if file already exists
            file_exists = os.path.isfile(filename)
            # Open the file in append mode if it exists, otherwise create a new 
            with open(filename, "a" if file_exists else "w") as file:
                # If the file exists, add a new line before writing the new data
                if file_exists:
                    file.write("\n")
                # Write the name and value as a comma-separated string to the file
                file.write(f"{criterion},{name},{sim}")