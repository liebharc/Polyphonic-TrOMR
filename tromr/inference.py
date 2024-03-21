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
        merged = merge_symbols(predrhythms, predpitchs, predlifts)
        return merged, predrhythms
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TrOMR Inference')
    parser.add_argument('image', type=str, help='Path to the input image')
    parser.add_argument('--output', type=str, default='output', help='Path to the output file')
    args = parser.parse_args()
    
    inference = Inference()
    pred, predrhythms = inference.predict(args.image)
    print(pred)
