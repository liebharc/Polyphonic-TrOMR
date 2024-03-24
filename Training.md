# Log of all training runs

- Optional: Install [NVidia Apex](https://github.com/NVIDIA/apex)
- Download and unpack [Camera-PrIMus](https://grfia.dlsi.ua.es/primus/) to `$gitroot/Corpus`
- Download and unpack [GrandStaff](https://sites.google.com/view/multiscore-project/datasets) to `$gitroot/grandstaff`
- Make sure you have installed pytorch and CUDA correctly

## Ideas

- Train with undistorted data

## Train log

### Run 1

Date: 19 Mar 2024
Training time: ~96h
Commit: 4e6d0ca9ced275ba4833df42628686c629bf072c
Result:

```
{'eval_loss': 0.0006982507184147835, 'eval_runtime': 371.581, 'eval_samples_per_second': 51.706, 'eval_steps_per_second': 12.929, 'epoch': 50.0}
{'train_runtime': 346761.4874, 'train_samples_per_second': 24.934, 'train_steps_per_second': 0.779, 'train_loss': 0.005856822736803091, 'epoch': 50.0}
Saved model to pytorch_model_1710495948.pth
```

A test run against a sheet music example shows that this model performs very poorly.

### Run 0

The weights from the [original paper](https://arxiv.org/abs/2308.09370).
