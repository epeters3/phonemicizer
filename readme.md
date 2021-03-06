# A seq2seq Phonemicizer

A sequence2sequence model that can translate lower case english words with no punctuation into their [arpabet](https://en.wikipedia.org/wiki/ARPABET) phonetical representations. Most of the code related to the seq2seq attention model itself is due thanks to [Sean Robertson's seq2seq+attention PyTorch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html).

The code that builds the training corpus is my own code. It takes NLTK's version of the CMU Pronunciation Dictionary and builds a training corpus of over 133,000 training instances. I also added a Levenshtein similarity metric that can be used to calculate model accuracy.

## Getting Started

To download the model:

```shell
git clone https://github.com/epeters3/phonemicizer.git
cd phonemicizer
pip3 install -r requirements.txt
```

## Predicting

To use the current best model saved in the repo to predict phonetic representations, there is a CLI and Python API.

### Using the CLI

The module can accept text via stdin, phonemicize it, then output to stdout, e.g.:

```shell
$ echo "Interesting." | python3 -m phonemicizer
N T ER AH S T IH NG
```

### Using the Python API

```python
from phonemicizer import TrainedPhonemicizer

model = TrainedPhonemicizer()
model.predict("Interesting.")
# N T ER AH S T IH NG
```

## Evaluation

To assess model performance by evaluating the trained model on some samples:

```shell
python3 -m phonemicizer.evaluate [--n <number_of_samples>] 
```


## Training

The repo includes state dictionaries for the model so it can go straight into predict and evaluate mode, but to train the model from scratch:

```shell
python3 -m phonemicizer.train \
    --n_iters <number_of_training_iterations> \
    [--epoch_length 1000] \
    [--plot_every 100] \
    [--learning_rate 0.0001] \
    [--weight_decay 0.0] \
    [--teacher_forcing_ratio 0.5] \
    [--dropout_p 0.1] \
    [--lr_patience 5] \
```

## Model Description

The model uses a single GRU layer in the encoder and an attention and GRU layer in the decoder, with a hidden size of 256 for all layers. Accuracy for the trained model weights currently in the repo is about 87% on the hold-out test set. The current model was trained for 120,000 iterations. 
