# Log of all training runs

- Optional: Install [NVidia Apex](https://github.com/NVIDIA/apex) - only needed if the `--fast` option is used during training
- Download and unpack [Camera-PrIMus](https://grfia.dlsi.ua.es/primus/) to `$gitroot/Corpus`
- Download and unpack [GrandStaff](https://sites.google.com/view/multiscore-project/datasets) to `$gitroot/grandstaff`
- Download and unpack [Documents in the Wild](https://github.com/cvlab-stonybrook/PaperEdge?tab=readme-ov-files) to `$gitroot/DIW`
- Clone [CPMS](https://github.com/itec-hust/CPMS) to `$gitroot/CPMS`
- Make sure you have installed pytorch and CUDA correctly

## To Checks

- Visualize attention: https://github.com/huggingface/pytorch-image-models/discussions/1232
- Different values for alpha and beta

## Train log

## Run 77 Increased depth

Date: 26 Apr 2024
Training time: ~19h (fast option)
Commit: f3658eb0a33cb0712ec980d944949e776e552989
SER: %

Encoder & decoder depth was increased from 4 to 6

## Run 76 Training data fix for accidentals

Date: 25 Apr 2024
Training time: ~16h (fast option)
Commit: 75d8688719494169f4b629fc51224d4aa846eee7
SER: 77%

Fixed that the training data didn't contain any natural accidentals.

## Run 74 Backtracking

Date: 24 Apr 2024
Training time: ~24h (fast option)
Commit: b4af54249fca5bf93650c518c7220f5de98c843c
SER: 77%

After experiments with focal loss and weight decay, we are backtracking to run 63.

## Run 74 

Date: 23 Apr 2024
Training time: ~24h (fast option)
Commit: 6580500e71602d5c74decde2946498c8e883392e
SER: 77%

Adding a weight to the lift/accidental tokens.

## Run 71 Weight decay

Date: 22 Apr 2024
Training time: ~17h (fast option)
Commit: 3b92eee2e56647fcb538b4ef5ef3704f12bfb2d1
SER: 77%

Reduced weight decay.

## Run 70 Focal loss

Date: 21 Apr 2024
Training time: ~17h (fast option), aborted after epoch 16 from 25
Commit: a6b87b71b3b69d87d424f3c86500081f6146d436
SER: 75%

Looks like a focal loss doesn't help to improve the performance of the lift detection.

## Run 63 Negative data set

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

### Run 0

The weights from the [original paper](https://arxiv.org/abs/2308.09370).
SER\*: 74%
Manual validation result: 9.3

- SER failues shown here should be taken with a grain of salt
