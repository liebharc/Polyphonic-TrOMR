import os

import argparse
import urllib.request
import cv2
import torch
import numpy as np
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from transformers import PreTrainedTokenizerFast

from model import TrOMR
from configs import default_config, Config

checkpoint_file_path = os.path.join(os.path.dirname(__file__), "workspace", "checkpoints", "img2score_epoch47.pth")

class Staff2Score(object):
    def __init__(self, config: Config):
        self.config = config
        self.size_h = config.max_height
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrOMR(config)
        if not os.path.exists(checkpoint_file_path):
            raise RuntimeError("Please download the model first to " + checkpoint_file_path)
        self.model.load_state_dict(torch.load(checkpoint_file_path), strict=False)
        self.model.to(self.device)
        
        if not os.path.exists(config.filepaths.rhythmtokenizer):
            raise RuntimeError("Failed to find tokenizer config" + config.filepaths.rhythmtokenizer)
        self.lifttokenizer = PreTrainedTokenizerFast(tokenizer_file=config.filepaths.lifttokenizer)
        self.pitchtokenizer = PreTrainedTokenizerFast(tokenizer_file=config.filepaths.pitchtokenizer)
        self.rhythmtokenizer = PreTrainedTokenizerFast(tokenizer_file=config.filepaths.rhythmtokenizer)
        self.transform = alb.Compose([
            alb.ToGray(always_apply=True),
            alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
            ToTensorV2(),
        ])
        
    def readimg(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if img.shape[-1] == 4:
            img = 255 - img[:,:,3]
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            raise RuntimeError("Unsupport image type!")
        
        h, w, c = img.shape
        new_h = self.size_h
        new_w = int(self.size_h / h * w)
        new_w = new_w // self.config.patch_size * self.config.patch_size
        img = cv2.resize(img, (new_w, new_h))
        img = self.transform(image=img)['image'][:1]
        return img
    
    def detokenize(self, tokens, tokenizer):
        toks = [tokenizer.convert_ids_to_tokens(tok) for tok in tokens]
        for b in range(len(toks)):
            for i in reversed(range(len(toks[b]))):
                if toks[b][i] is None:
                    toks[b][i] = ''
                toks[b][i] = toks[b][i].replace('Ġ', ' ').strip()
                if toks[b][i] in (['[BOS]', '[EOS]', '[PAD]']):
                    del toks[b][i]
        return toks
    
    def predict_token(self, imgpath):
        imgs = []
        if os.path.isdir(imgpath):
            for item in os.listdir(imgpath):
                imgs.append(self.readimg(os.path.join(imgpath, item)))
        else:
            imgs.append(self.readimg(imgpath))
        imgs = torch.cat(imgs).float().unsqueeze(1)
        output = self.model.generate(imgs.to(self.device),
                                    temperature=self.config.temperature)
        rhythm, pitch, lift = output
        return rhythm, pitch, lift

    def predict(self, imgpath):
        rhythm, pitch, lift = self.predict_token(imgpath)
        
        predlift = self.detokenize(lift, self.lifttokenizer)
        predpitch = self.detokenize(pitch, self.pitchtokenizer)
        predrhythm = self.detokenize(rhythm, self.rhythmtokenizer)
        return predrhythm, predpitch, predlift