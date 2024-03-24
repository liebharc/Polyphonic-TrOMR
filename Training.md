# Log of all training runs

- Optional: Install [NVidia Apex](https://github.com/NVIDIA/apex) - only needed if the `--fast` option is used during training
- Download and unpack [Camera-PrIMus](https://grfia.dlsi.ua.es/primus/) to `$gitroot/Corpus`
- Download and unpack [GrandStaff](https://sites.google.com/view/multiscore-project/datasets) to `$gitroot/grandstaff`
- Make sure you have installed pytorch and CUDA correctly

## Train log

### Run1

Date: 24 Mar 2024
Training time: ~24h (fast option)
Commit: 74adcbc396b462c77ad8d1fd8f4f3c6a01b5836e
Result:

```
{'eval_loss': 0.003167663002386689, 'eval_runtime': 168.9123, 'eval_samples_per_second': 111.022, 'eval_steps_per_second': 13.883, 'epoch': 40.0}
{'train_runtime': 86163.9708, 'train_samples_per_second': 78.355, 'train_steps_per_second': 4.897, 'train_loss': 0.04649836303557555, 'epoch': 40.0}
```

### Run 0

The weights from the [original paper](https://arxiv.org/abs/2308.09370).
