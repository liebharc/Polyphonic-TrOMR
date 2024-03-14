import cv2
import numpy as np
import random
import json
import os
import re
from typing import List

import torch
from configs import Config
from split_merge_symbols import split_symbols

script_location = os.path.dirname(os.path.realpath(__file__))

os.environ["WANDB_DISABLED"] = "true"

def normalize(image):
    return (255. - image)/255.

def resize(image, height):
    width = int(float(height * image.shape[1]) / image.shape[0])
    sample_img = cv2.resize(image, (width, height))
    return sample_img


def add_image_into_tr_omr_canvas(image: np.ndarray, margin_top: int, margin_bottom: int) -> np.ndarray:
    tr_omr_max_height=128
    tr_omr_max_width=1280
    tr_omr_max_height_with_margin = tr_omr_max_height - margin_top - margin_bottom
    tr_omr_ratio = float(tr_omr_max_height_with_margin) / tr_omr_max_width
    height, width = image.shape[:2]

    # Calculate the new size such that it fits exactly into the tr_omr_max_height and tr_omr_max_width
    # while maintaining the aspect ratio of height and width.
    
    if height / width > tr_omr_ratio:
        # The height is the limiting factor.
        new_shape = (int(width / height * tr_omr_max_height_with_margin), tr_omr_max_height_with_margin)
    else:
        # The width is the limiting factor.
        new_shape = (tr_omr_max_width, int(height / width * tr_omr_max_width))
       
    resized = cv2.resize(image, new_shape)
    
    new_image = np.zeros((tr_omr_max_height, tr_omr_max_width), np.uint8)
    new_image[:, :] = 255
    
    # Copy the resized image into the center of the new image.
    x_offset = 0
    y_offset = (tr_omr_max_height_with_margin - resized.shape[0]) // 2 + margin_top
    new_image[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized

    return new_image, new_shape[0] / width

git_root = os.path.join(script_location, '..')

class CTC_PriMuS():
    """
    Dataset class for the CTC PriMuS dataset and all datasets which have been preprocessed to have the same format.

    The format is an image file and a semantic file. The semantic file contains the ground truth.
    """

    gt_element_separator = '-'
    PAD_COLUMN = 0
    validation_dict = None


    def __init__(self, corpus_list, rhythm_vocab, pitch_vocab, note_vocab, lift_vocab, distortions, config: Config):
        self.current_idx = 0
        self.distortions = distortions
        self.corpus_list = corpus_list
        self.rhythm_vocab = rhythm_vocab
        self.pitch_vocab = pitch_vocab
        self.note_vocab = note_vocab
        self.lift_vocab = lift_vocab
        self.config = config

    def __len__(self):
        return len(self.corpus_list)
    
    def _get_x_center_from_box(self, box):
        return (box[0][0] + box[1][0]) / 2.0
    
    def _pad_rhythm(self, samples):
        if len(samples) > self.config.max_seq_len - 2:
            samples = samples[:self.config.max_seq_len - 2]
        samples.append(self.config.eos_token)
        samples.insert(0, self.config.bos_token)
        samples_padded = np.ones(self.config.max_seq_len, dtype=np.int32) * self.PAD_COLUMN
        valid_len = min(self.config.max_seq_len, len(samples))
        samples_padded[:valid_len] = np.array(samples[:valid_len])
        return samples_padded
    
    def _pad_samples(self, samples):
        if len(samples) > self.config.max_seq_len:
            samples = samples[:self.config.max_seq_len]
        samples.append(self.config.nonote_token)
        samples.insert(0, self.config.nonote_token)
        samples_padded = np.ones(self.config.max_seq_len, dtype=np.int32) * self.PAD_COLUMN
        valid_len = min(self.config.max_seq_len, len(samples))
        samples_padded[:valid_len] = np.array(samples[:valid_len])
        return samples_padded
    
    def _check_seq_values(self, seq, max_value):
        for value in seq:
            if value >= max_value or value < 0:
                raise Exception('ERROR: ' + str(value) + ' not in range of 0 to ' + str(max_value) + '!')
        return seq
    
    def __getitem__(self, idx):
        sample_filepath = self.corpus_list[idx]
        sample_fullpath = os.path.join(git_root, sample_filepath.split(".")[0].replace("_distorted", ""))

        # IMAGE
        if self.distortions:
            sample_img = cv2.imread(sample_fullpath + '_distorted.jpg', cv2.IMREAD_GRAYSCALE) # Grayscale is assumed
        elif os.path.exists(sample_fullpath + '.png'):
            sample_img = cv2.imread(sample_fullpath + '.png', cv2.IMREAD_GRAYSCALE)  # Grayscale is assumed!
        else:
            sample_img = cv2.imread(sample_fullpath + '.jpg', cv2.IMREAD_GRAYSCALE)
        height = self.config.max_height
        #sample_img = resize(sample_img,height)
        # Resizing is wrong, instead center and surround with white
        width = self.config.max_width
        margin_top = random.randint(0, 13)
        margin_bottom = random.randint(0, 13)
        sample_img, x_ratio = add_image_into_tr_omr_canvas(sample_img, margin_top, margin_bottom)
        image = normalize(sample_img)
        image_norm = np.zeros((1, height, width)).astype(np.float32)
        image_norm[0] = image.astype(np.float32)

        # GROUND TRUTH
        sample_full_filepath = sample_fullpath + '.semantic'
        
        sample_gt_file = open(sample_full_filepath, 'r')
        sample_gt_plain = sample_gt_file.readline().rstrip()
        sample_gt_file.close()

        liftsymbols, pitchsymbols, rhythmsymbols, note_symbols = split_symbols([sample_gt_plain])

        rhythm = _translate_symbols(rhythmsymbols[0], self.rhythm_vocab, self.config.pad_token, 'rhythm')
        lifts = _translate_symbols(liftsymbols[0], self.lift_vocab, self.config.nonote_token, 'lift')
        pitch = _translate_symbols(pitchsymbols[0], self.pitch_vocab, self.config.nonote_token, 'pitch')
        notes = _translate_symbols(note_symbols[0], self.note_vocab, self.config.nonote_token, 'note')
        mask = np.ones_like(image_norm).astype(np.bool_)
        return {
            'inputs': image_norm,
            'mask': mask,
            'rhythms_seq': self._check_seq_values(self._pad_rhythm(rhythm), self.config.num_rhythm_tokens),
            'note_seq': self._check_seq_values(self._pad_samples(notes), self.config.num_note_tokens),
            'lifts_seq': self._check_seq_values(self._pad_samples(lifts), self.config.num_lift_tokens),
            'pitchs_seq': self._check_seq_values(self._pad_samples(pitch), self.config.num_pitch_tokens),
        }
    
def _translate_symbols(symbols, vocab, default_token, vocab_name):
    result = []
    for symbol in symbols:
        if symbol in vocab:
            result.append(vocab[symbol])
        else:
            print('Warning: ' + symbol + ' not in ' + vocab_name + ' vocabulary')
            result.append(default_token)
    return result


class DataCollator:
    def __call__(self, examples: List[dict], return_tensors="pt"):
        return {
            "inputs": torch.tensor(np.stack([f["inputs"] for f in examples])),
            "mask": torch.tensor(np.stack([f["mask"] for f in examples]), dtype=torch.bool),
            "rhythms_seq": torch.tensor(np.stack([f["rhythms_seq"] for f in examples]), dtype=torch.long),
            "note_seq": torch.tensor(np.stack([f["note_seq"] for f in examples]), dtype=torch.long),
            "lifts_seq": torch.tensor(np.stack([f["lifts_seq"] for f in examples]), dtype=torch.long),
            "pitchs_seq": torch.tensor(np.stack([f["pitchs_seq"] for f in examples]), dtype=torch.long),
        }

def _translate_note_to_rhythm(note: str) -> str:
    # Define the regex pattern for the note part
    pattern = r"note-[A-G][#b]?[0-9]_"

    # Replace the note part with 'note-'
    translated_note = re.sub(pattern, "note-", note)
    translated_note = translated_note.replace("_fermata", "")
    translated_note = translated_note.replace("..", ".")
    translated_note = translated_note.replace("grace", "")
    translated_note = translated_note.replace("double_whole", "whole")
    translated_note = translated_note.replace("quadruple_whole", "whole")

    return translated_note

def _translate_multirests(note: str, tokenizer_vocab) -> str:
    if not note.startswith('multirest'):
        return note
    if note == 'multirest-1':
        return 'multirest-2'
    if note not in tokenizer_vocab:
        return "multirest-98"
    return note

def _translate_time_signature(note: str, tokenizer_vocab) -> str:
    if not note.startswith('timeSignature'):
        return note
    if note not in tokenizer_vocab:
        return "timeSignature-4/4"
    return note


rhythm_tokenizer_path = os.path.join(script_location, 'workspace', 'tokenizers', 'tokenizer_rhythm.json')
pitch_tokenizer_path = os.path.join(script_location,  'workspace', 'tokenizers', 'tokenizer_pitch.json')
note_tokenizer_path = os.path.join(script_location,  'workspace', 'tokenizers', 'tokenizer_note.json')
lift_tokenizer_path = os.path.join(script_location,  'workspace', 'tokenizers', 'tokenizer_lift.json')

def load_primus(samples, 
                config, 
                distortions = False, 
                val_split = 0.0):
    rhythm_tokenizer_config = json.load(open(rhythm_tokenizer_path,'r'))
    pitch_tokenizer_config = json.load(open(pitch_tokenizer_path,'r'))
    note_tokenizer_config = json.load(open(note_tokenizer_path,'r'))
    lift_tokenizer_config = json.load(open(lift_tokenizer_path,'r'))

    rhythm_tokenizer_vocab = rhythm_tokenizer_config['model']['vocab']
    pitch_tokenizer_vocab = pitch_tokenizer_config['model']['vocab']
    note_tokenizer_vocab = note_tokenizer_config['model']['vocab']
    lift_tokenizer_vocab = lift_tokenizer_config['model']['vocab']
    
    # Train and validation split
    val_idx = int(len(samples) * val_split) 
    training_list = samples[val_idx:]
    validation_list = samples[:val_idx]
    
    print ('Training with ' + str(len(training_list)) + ' and validating with ' + str(len(validation_list)))
    return {
        "train": CTC_PriMuS(training_list, rhythm_tokenizer_vocab, pitch_tokenizer_vocab, note_tokenizer_vocab, lift_tokenizer_vocab, distortions, config),
        "validation": CTC_PriMuS(validation_list, rhythm_tokenizer_vocab, pitch_tokenizer_vocab, note_tokenizer_vocab, lift_tokenizer_vocab, distortions, config),
    }

