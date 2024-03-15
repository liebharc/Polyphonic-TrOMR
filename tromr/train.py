import torch
import numpy as np
import os
from pathlib import Path
import sys
import shutil
from typing import List
import time

from transformers import TrainingArguments, Trainer
from primus import CTC_PriMuS, load_primus, DataCollator
from model.tromr_arch import TrOMR
from configs import default_config
from convert_grandstaff import convert_grandstaff
from mix_datasets import mix_training_sets

script_location = os.path.dirname(os.path.realpath(__file__))

vocabulary = os.path.join(script_location, 'vocabulary_semantic.txt')
git_root = os.path.join(script_location, '..')
primus = os.path.join(git_root, 'Corpus')
grandstaff = os.path.join(git_root, 'grandstaff')
cpms = os.path.join(git_root, 'CPMS')
primus_train_index = os.path.join(primus, 'index.txt')
cpms_train_index = os.path.join(git_root, 'cpms_index.txt')
grandstaff_train_index = os.path.join(grandstaff, 'index.txt')

tr_omr_pretrained = os.path.join(script_location, 'workspace', 'checkpoints', 'img2score_epoch47.pth')

number_of_files = -1
number_of_epochs = 50

def index_primus_dataset():
    print('Indexing Primus dataset')
    with open(primus_train_index, 'w') as f:
        for path in Path(primus).rglob('*distorted.jpg'):
            f.write(str(path.relative_to(git_root)) + '\n')
    print('Done indexing')

if not os.path.exists(primus_train_index):
    index_primus_dataset()

def index_cpms_dataset():
    print('Indexing CPMS dataset')
    with open(cpms_train_index, 'w') as f:
        for path in Path(os.path.join(cpms, "semantic", "training")).rglob('*.jpg'):
            semantic = str(path.relative_to(git_root)).replace(".jpg", ".semantic")
            if os.path.exists(semantic):
                f.write(str(path.relative_to(git_root)) + '\n')
    print('Done indexing')

if not os.path.exists(primus_train_index):
    index_primus_dataset()

if not os.path.exists(cpms_train_index):
    index_cpms_dataset()

if not os.path.exists(grandstaff_train_index):
    convert_grandstaff()

if os.path.exists(os.path.join(git_root, "test_primus")):
    shutil.rmtree(os.path.join(git_root, "test_primus"))

def load_training_index(file_path: str):
    with open(file_path, 'r') as f:
        return f.readlines()

    
def load_and_mix_training_sets(index_paths: List[str], weights: List[float], number_of_files: int):
    data_sources = [load_training_index(index) for index in index_paths]
    print("Total number of training files to choose from", sum([len(data) for data in data_sources]))
    return mix_training_sets(data_sources, weights, number_of_files)

train_index = load_and_mix_training_sets([primus_train_index, cpms_train_index, grandstaff_train_index], [1.0, 1.0, 1.0], number_of_files)

datasets = load_primus(train_index, default_config, val_split = 0.1, distortions=True)
    
data_collator = DataCollator()

args = TrainingArguments(
    f"test-primus",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=1e-3,
    optim="adamw_torch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=4,
    num_train_epochs=number_of_epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    logging_dir='logs',
    save_strategy="epoch",
    label_names=['rhythms_seq', 'note_seq', 'lifts_seq', 'pitchs_seq'],
)

if len(sys.argv) > 1 and sys.argv[1] == '--pretrained':
    print('Loading pretrained model')
    model = TrOMR(default_config)
    model.load_state_dict(torch.load(tr_omr_pretrained), strict=False)
else:
    model = TrOMR(default_config)


timestamp = str(round(time.time()))
try:
    trainer = Trainer(
        model,
        args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
    )

    trainer.train()
except KeyboardInterrupt:
    print('Interrupted')

model_destination = os.path.join(script_location, 'workspace', 'checkpoints', f'pytorch_model_{timestamp}.pth')
torch.save(model.state_dict(), model_destination)
print(f'Saved model to {model_destination}')