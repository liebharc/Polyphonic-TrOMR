import numpy as np
import json
import os

import torch
from configs import Config
from split_merge_symbols import split_symbols
from image_processing import readimg

script_location = os.path.dirname(os.path.realpath(__file__))

os.environ["WANDB_DISABLED"] = "true"

git_root = os.path.join(script_location, '..')

class DataLoader():
    """
    Dataset class for the CTC PriMuS dataset and all datasets which have been preprocessed to have the same format.

    The format is an image file and a semantic file. The semantic file contains the ground truth.
    """

    gt_element_separator = '-'
    PAD_COLUMN = 0
    validation_dict = None


    def __init__(self, corpus_list, rhythm_vocab, pitch_vocab, note_vocab, lift_vocab, config: Config):
        self.current_idx = 0
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
    
    def _limit_samples(self, samples):
        if len(samples) > self.config.max_seq_len - 2:
            samples = samples[:self.config.max_seq_len - 2]
        return samples
    
    def _pad_array_to_max_seq_len(self, samples):
        samples_padded = np.ones(self.config.max_seq_len, dtype=np.int64) * self.PAD_COLUMN
        valid_len = min(self.config.max_seq_len, len(samples))
        samples_padded[:valid_len] = np.array(samples[:valid_len])
        #tensor = torch.from_numpy(samples_padded).to(cuda0)
        return samples_padded
    
    def _pad_rhythm(self, samples):
        samples = self._limit_samples(samples)
        samples.append(self.config.eos_token)
        samples.insert(0, self.config.bos_token)
        return samples
    
    def _pad_samples(self, samples):
        samples = self._limit_samples(samples)
        samples.append(self.config.nonote_token)
        samples.insert(0, self.config.nonote_token)
        return samples
    
    def _check_seq_values(self, seq, max_value):
        for value in seq:
            if value >= max_value or value < 0:
                raise Exception('ERROR: ' + str(value) + ' not in range of 0 to ' + str(max_value) + '!')
        return seq
    
    def _check_index(self, idx):
        try:
            self[idx]
            return True
        except Exception as e:
            print('ERROR: ' + str(e))
            return False
    
    def check(self):
        """
        Loads every entry to check if the files are available
        and can be loaded correctly.
        """
        has_errors = False
        i = 0
        
        for i in range(len(self)):
            result = self._check_index(i)
            has_errors = has_errors or not result
            i += 1
            if i % 10000 == 0:
                print('Checked ' + str(i) + '/' + str(len(self)) + ' entries')
        return has_errors
    
    def __getitem__(self, idx):
        entry = self.corpus_list[idx].strip()
        sample_filepath = entry.split(",")[0]
        sample_img = readimg(self.config, os.path.join(git_root, sample_filepath))

        # GROUND TRUTH
        sample_full_filepath = entry.split(",")[1]

        sample_gt_file = open(os.path.join(git_root, sample_full_filepath), 'r')
        sample_gt_plain = sample_gt_file.readline().rstrip()
        sample_gt_file.close()

        liftsymbols, pitchsymbols, rhythmsymbols, note_symbols = split_symbols([sample_gt_plain])

        rhythm = _translate_symbols(rhythmsymbols[0], self.rhythm_vocab, self.config.pad_token, 'rhythm')
        lifts = _translate_symbols(liftsymbols[0], self.lift_vocab, self.config.nonote_token, 'lift')
        pitch = _translate_symbols(pitchsymbols[0], self.pitch_vocab, self.config.nonote_token, 'pitch')
        notes = _translate_symbols(note_symbols[0], self.note_vocab, self.config.nonote_token, 'note')
        rhythm_seq = self._check_seq_values(self._pad_rhythm(rhythm), self.config.num_rhythm_tokens)
        mask = np.zeros(self.config.max_seq_len).astype(np.bool_)
        mask[:len(rhythm_seq)] = 1
        result = {
            'inputs': sample_img,
            'mask': mask,
            'rhythms_seq': self._pad_array_to_max_seq_len(rhythm_seq),
            'note_seq': self._pad_array_to_max_seq_len(self._check_seq_values(self._pad_samples(notes), self.config.num_note_tokens)),
            'lifts_seq': self._pad_array_to_max_seq_len(self._check_seq_values(self._pad_samples(lifts), self.config.num_lift_tokens)),
            'pitchs_seq': self._pad_array_to_max_seq_len(self._check_seq_values(self._pad_samples(pitch), self.config.num_pitch_tokens)),
        }
        return result
    
def _translate_symbols(symbols, vocab, default_token, vocab_name):
    result = []
    for symbol in symbols:
        if symbol in vocab:
            result.append(vocab[symbol])
        else:
            print('Warning: ' + symbol + ' not in ' + vocab_name + ' vocabulary')
            result.append(default_token)
    return result


rhythm_tokenizer_path = os.path.join(script_location, 'workspace', 'tokenizers', 'tokenizer_rhythm.json')
pitch_tokenizer_path = os.path.join(script_location,  'workspace', 'tokenizers', 'tokenizer_pitch.json')
note_tokenizer_path = os.path.join(script_location,  'workspace', 'tokenizers', 'tokenizer_note.json')
lift_tokenizer_path = os.path.join(script_location,  'workspace', 'tokenizers', 'tokenizer_lift.json')

def load_dataset(samples, 
                 config, 
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
        "train": DataLoader(training_list, rhythm_tokenizer_vocab, pitch_tokenizer_vocab, note_tokenizer_vocab, lift_tokenizer_vocab, config),
        "train_list": training_list,
        "validation": DataLoader(validation_list, rhythm_tokenizer_vocab, pitch_tokenizer_vocab, note_tokenizer_vocab, lift_tokenizer_vocab, config),
        "validation_list": validation_list
    }

