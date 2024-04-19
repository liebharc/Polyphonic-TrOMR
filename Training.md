# Log of all training runs

- Optional: Install [NVidia Apex](https://github.com/NVIDIA/apex) - only needed if the `--fast` option is used during training
- Download and unpack [Camera-PrIMus](https://grfia.dlsi.ua.es/primus/) to `$gitroot/Corpus`
- Download and unpack [GrandStaff](https://sites.google.com/view/multiscore-project/datasets) to `$gitroot/grandstaff`
- Download and unpack [Documents in the Wild](https://github.com/cvlab-stonybrook/PaperEdge?tab=readme-ov-files) to `$gitroot/DIW`
- Clone [CPMS](https://github.com/itec-hust/CPMS) to `$gitroot/CPMS`
- Make sure you have installed pytorch and CUDA correctly

## To Checks

- Label Smoothing
- Increase data set by fully masking
- Visualize attention: https://github.com/huggingface/pytorch-image-models/discussions/1232
- Different values for alpha and beta
- Restore additional rhythm classes
- Dataset with negative exampkes (no sheet music)

## Train log

## Run 69

Date: 20 Apr 2024
Training time: ~15h (fast option), aborted after epoch 14 from 25
Commit: 3f524a8c291c32eb19998facef77c9e12599a2a8
SER: 73%
Manual validation result: 16.7

## Run 68

Date: 14 Apr 2024
Training time: ~8h (fast option), aborted after epoch 8 from 25
Commit: fb86d320fe82c81c352b69a059024a883e83c346
SER: 78%

## Run 67

Date: 14 Apr 2024
Training time: ~24h
Commit: fb86d320fe82c81c352b69a059024a883e83c346
SER: 76%

## Run 63

Date: 11 Apr 2024
Training time: ~26h (fast option)
Commit: c360ab726df18879973e6829a1423c627a99afd5
SER: 74%
Manual validation result: 13.7

Increased data set size by introducing a negative data set with no musical symbols. And by using positive data sets more often with different mask values.

## Run 57 Dropout 0.8

Date: 07 Apr 2024
Training time: ~14h (fast option)
Commit: 3fc893c0ab547fe1958adf500b0afaf0f6990f80
SER: 81%

Changes to the conversion of the grandstaff dataset haven't been applied yet.

## Run 56 Dropout 0.1

Date: 07 Apr 2024
Training time: ~14h (fast option)
Commit: 5ec6beaf461c034340ad0d2f832d842bef8bee75
SER: 72%
Manual validation result: 13.8

Changes to the conversion of the grandstaff dataset haven't been applied yet.

## Run 55 Dropout 0.2

Date: 06 Apr 2024
Training time: ~14h (fast option)
Commit: d73d5a9d342d4d934c21409632f4e2854d14d333
SER: 74%
Manual validation result: 17.0

Changes to the conversion of the grandstaff dataset haven't been applied yet.

## Run 51 Dropout 0

Start of dropout tests, number ranges for dropouts are mainly based on https://arxiv.org/pdf/2303.01500.pdf.

Date: 05 Apr 2024
Training time: ~14h (fast option)
Commit: cd445caa5337d86cf723854cb2ef9e98dd4c5b76
SER: 72%
Manual validation result: 18.4

## Run50 InceptionResnetV2

We changed how we number runs and established a link between the run number and the git history.

Date: 05 Apr 2024
Training time: ~19h (fast option)
Commit: a57ee4c046842c0135adca84f06260cff8af732f
SER: 88%

We tried InceptionResnetV2. The training run showed overfitting and the resulting SER indicates poor results. The model is over 3 times larger than the ResNetV2 model and might require more work to prevent overfitting.

### Run3

Date: 02 Apr 2024
Training time: ~24h (fast option)
Commit: 9ddfff8b5782473e8831ca3791d9bef99f726654
SER: 73%
Manual validation result: 23.4

We decreased the vocabulary, the alpha/beta ratio in the loss function and made changes to the grandstaff dataset. While still performing worse than Run 0 in the manual validation, it gets closer now and in some specific tests performs even better than Run 0. We will have to backtrack from this point to find out which of the changes lead to an improved result.

### Run2

Date: 01 Apr 2024
Training time: ~48h
Commit: 516093a3f3840cb82922b4d7300d1568455277d568f85ea96fe41235a06ca8de6759f1db6b8fc39a
SER: 79%

### Run1

Date: 24 Mar 2024
Training time: ~24h (fast option)
Commit: 516093a3f3841235a06ca8de6759f1db6b8fc39a
SER: 82%
Result:

```
{'eval_loss': 0.003167663002386689, 'eval_runtime': 168.9123, 'eval_samples_per_second': 111.022, 'eval_steps_per_second': 13.883, 'epoch': 40.0}
{'train_runtime': 86163.9708, 'train_samples_per_second': 78.355, 'train_steps_per_second': 4.897, 'train_loss': 0.04649836303557555, 'epoch': 40.0}
```

### Run 0

The weights from the [original paper](https://arxiv.org/abs/2308.09370).
SER\*: 74%
Manual validation result: 9.3

- SER failues shown here should be taken with a grain of salt
