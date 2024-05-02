import os

import torch
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from transformers import PreTrainedTokenizerFast

from model import TrOMR
from configs import Config
from image_processing import readimg
import safetensors

default_checkpoint_file_path = os.path.join(os.path.dirname(__file__), "workspace", "checkpoints", "pytorch_model_82-df5dbfdb19ea8a18b0d0c2df2efa40a04abd6206.pth")

class Staff2Score(object):
    def __init__(self, config: Config, checkpoint_file_path = default_checkpoint_file_path):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrOMR(config)
        if not os.path.exists(checkpoint_file_path):
            raise RuntimeError("Please download the model first to " + checkpoint_file_path)
        if ".safetensors" in checkpoint_file_path:
            tensors = {}
            with safetensors.safe_open(checkpoint_file_path, framework="pt", device=0) as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
            self.model.load_state_dict(tensors, strict=False)
        else:
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
                imgs.append(readimg(self.config, os.path.join(imgpath, item)))
        else:
            imgs.append(readimg(self.config, imgpath))
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