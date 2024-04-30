import numpy as np
import json
import os

import torch
import random
from configs import Config
from split_merge_symbols import split_semantic_file
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
        self.corpus_list = self._add_mask_steps(corpus_list)
        self.rhythm_vocab = rhythm_vocab
        self.pitch_vocab = pitch_vocab
        self.note_vocab = note_vocab
        self.lift_vocab = lift_vocab
        self.config = config

    def _add_mask_steps(self, corpus_list):
        result = []
        for entry in corpus_list:
            image, semantic_file = entry.strip().split(',')
            semantic = self._read_semantic(semantic_file)
            lifts = semantic[0][0]
            semantic_len = len(lifts)
            number_of_flats = lifts.count("lift_b")
            number_of_sharps = lifts.count("lift_#")
            number_of_naturals = lifts.count("lift_N")
            # If we would have the money to do it we would want to use: mask_lens = range(1, semantic_len)
            # Instead we construct take up to 3 random mask lengths and the full length
            mask_lens = set()
            mask_lens.add(semantic_len + 2)
            image_basename = os.path.basename(image)
            image_basename_hash = hash(image_basename)

            # Seed the random generator here to get more reproducible results
            # Although it isn't clear if this really affects the results more 
            # than all the randomness in the conversion of the data sets
            random.seed(image_basename_hash)
            number_of_desired_samples = 2
            if number_of_naturals > number_of_flats + number_of_sharps:
                # If we have more naturals than other accidentals than the naturals
                # likely are there to cancel the key signature. Since we don't have
                # a lot of examples for this we create additional data
                number_of_desired_samples *= 2
            for _ in range(1, min(number_of_desired_samples, semantic_len)):
                mask_lens.add(random.randint(1, semantic_len) + 1)
            
            #mask_lens = range(2, semantic_len + 2)  # + 2 for the BOS and EOS token
            for mask_len in mask_lens:
                result.append({
                    "image": image,
                    "semantic": semantic_file,
                    "mask_len": mask_len
                })
        return result

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
    
    def _read_semantic(self, path):
        if path == "nosymbols":
            return [[]], [[]], [[]], [[]]
        return split_semantic_file(path)
    
    def __getitem__(self, idx):
        entry = self.corpus_list[idx]
        sample_filepath = entry["image"]
        sample_img = readimg(self.config, os.path.join(git_root, sample_filepath))

        # ground truth
        sample_full_filepath = entry["semantic"]
        liftsymbols, pitchsymbols, rhythmsymbols, note_symbols = self._read_semantic(sample_full_filepath)

        rhythm = _translate_symbols(rhythmsymbols[0], self.rhythm_vocab, self.config.pad_token, 'rhythm')
        lifts = _translate_symbols(liftsymbols[0], self.lift_vocab, self.config.nonote_token, 'lift')
        pitch = _translate_symbols(pitchsymbols[0], self.pitch_vocab, self.config.nonote_token, 'pitch')
        notes = _translate_symbols(note_symbols[0], self.note_vocab, self.config.nonote_token, 'note')
        rhythm_seq = self._check_seq_values(self._pad_rhythm(rhythm), self.config.num_rhythm_tokens)
        mask = np.zeros(self.config.max_seq_len).astype(np.bool_)
        mask[:entry["mask_len"]] = 1
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
pitch_tokenizer_path = os.path.join(script_location, 'workspace', 'tokenizers', 'tokenizer_pitch.json')
note_tokenizer_path = os.path.join(script_location, 'workspace', 'tokenizers', 'tokenizer_note.json')
lift_tokenizer_path = os.path.join(script_location, 'workspace', 'tokenizers', 'tokenizer_lift.json')

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

