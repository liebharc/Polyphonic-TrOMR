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
from staff2score import Staff2Score
from split_merge_symbols import merge_symbols

class Inference:
    def __init__(self):
        self.config = default_config
        self.handler = Staff2Score(self.config)

    def predict(self, filepath: str):
        predrhythms, predpitchs, predlifts = self.handler.predict(filepath)
        print(predrhythms)
        print(len(predrhythms[0]), len(predpitchs[0]), len(predlifts[0]))
        print([p for p in predlifts[0] if p != "lift_null" and p != "nonote"])
        print([p for p in predpitchs[0] if p != "lift_null" and p != "nonote"])
    
        merged = merge_symbols(predrhythms[0], predpitchs[0], predlifts[0])
        return merged, predrhythms
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TrOMR Inference')
    parser.add_argument('image', type=str, help='Path to the input image')
    parser.add_argument('--output', type=str, default='output', help='Path to the output file')
    args = parser.parse_args()
    
    inference = Inference()
    pred, predrhythms = inference.predict(args.image)
    print(pred)
