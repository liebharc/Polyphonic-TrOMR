import argparse
import torch
import os
import sys
import shutil
from typing import List
import time

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from transformers import TrainingArguments, Trainer
from data_loader import load_dataset
from model.tromr_arch import TrOMR
from configs import Config
from convert_grandstaff import grandstaff_train_index, convert_grandstaff
from convert_primus import primus_train_index, convert_primus_dataset, primus_distorted_train_index, cpms_train_index, convert_cpms_dataset
from convert_documents_in_the_wild import diw_train_index, convert_diw_dataset
from mix_datasets import mix_training_sets


def load_training_index(file_path: str):
    with open(file_path, 'r') as f:
        return f.readlines()


def check_data_source(all_file_paths: List[str]):
    result = True
    for file_paths in all_file_paths:
        paths = file_paths.strip().split(",")
        for path in paths:
            if path == "nosymbols":
                continue
            if not os.path.exists(path):
                print(f'Index {file_paths} does not exist due to {path}')
                result = False
    return result

    
def load_and_mix_training_sets(index_paths: List[str], weights: List[float], number_of_files: int):
    data_sources = [load_training_index(index) for index in index_paths]
    if not all([check_data_source(data) for data in data_sources]):
        print('Error in datasets found')
        sys.exit(1)
    print("Total number of training files to choose from", sum([len(data) for data in data_sources]))
    return mix_training_sets(data_sources, weights, number_of_files)


script_location = os.path.dirname(os.path.realpath(__file__))

vocabulary = os.path.join(script_location, 'vocabulary_semantic.txt')
git_root = os.path.join(script_location, '..')

tr_omr_pretrained = os.path.join(script_location, 'workspace', 'checkpoints', 'img2score_epoch47.pth')

number_of_files = -1
number_of_epochs = 20

parser = argparse.ArgumentParser(
                prog='train.py',
                description='TrOMR training')

parser.add_argument('--resume', help='Resume training from a checkpoint')
parser.add_argument('--pretrained', help='Resume training from current weights', action="store_true")
parser.add_argument("--fast", help='Uses options to accelerate the training process', action="store_true")

resume_from_checkpoint = None

args = parser.parse_args()

checkpoint_folder = "current_training"
if args.resume:
    resume_from_checkpoint = args.resume
else:
    if os.path.exists(os.path.join(git_root, checkpoint_folder)):
        shutil.rmtree(os.path.join(git_root, checkpoint_folder))


if not os.path.exists(primus_train_index) or not os.path.exists(primus_distorted_train_index):
    convert_primus_dataset()

if not os.path.exists(cpms_train_index):
    convert_cpms_dataset()

if not os.path.exists(grandstaff_train_index):
    convert_grandstaff()

if not os.path.exists(diw_train_index):
    convert_diw_dataset()

train_index = load_and_mix_training_sets([primus_train_index, cpms_train_index, grandstaff_train_index, diw_train_index], [1.0, 1.0, 1.0, 1.0], number_of_files)

config = Config()
if args.fast:
    optim = "adamw_apex_fused"
    config.reduced_precision = True
else:
    optim = "adamw_torch"  # TrOMR Paper page 3 species an Adam optimizer

datasets = load_dataset(train_index, config, val_split = 0.1)

print(f'Using {optim} optimizer')
compile_model = number_of_files < 0 or number_of_files * number_of_epochs >= 50000  # Compiling needs time, but pays off for large datasets
if compile_model:
    print('Compiling model')

git_count = os.popen('git rev-list --count HEAD').read().strip()
git_head = os.popen('git rev-parse HEAD').read().strip()

train_args = TrainingArguments(
    checkpoint_folder,
    torch_compile=compile_model,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=1e-4,  # TrOMR Paper page 3 specifies 1e-3, but that can cause issues with fp16 mode
    optim=optim,  
    per_device_train_batch_size=16,  # TrOMR Paper page 3
    per_device_eval_batch_size=8,
    num_train_epochs=number_of_epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    logging_dir=os.path.join('logs', f'run{git_count}-{git_head}'),
    save_strategy="epoch",
    label_names=['rhythms_seq', 'note_seq', 'lifts_seq', 'pitchs_seq'],
    fp16=args.fast,
    dataloader_pin_memory=True,
    dataloader_num_workers=12,
)

if args.pretrained:
    print('Loading pretrained model')
    model = TrOMR(config)
    model.load_state_dict(torch.load(tr_omr_pretrained), strict=False)
else:
    model = TrOMR(config)

try:
    trainer = Trainer(
        model,
        train_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
except KeyboardInterrupt:
    print('Interrupted')

model_destination = os.path.join(script_location, 'workspace', 'checkpoints', f'pytorch_model_{git_count}-{git_head}.pth')
torch.save(model.state_dict(), model_destination)
print(f'Saved model to {model_destination}')