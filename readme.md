# A seq2seq Phonemicizer

A sequence2sequence model that can translate lower case english words with not punctuation into their arpabet phonetical representations. Most of the code related to the seq2seq attention model itself is due thanks to [Sean Robertson's seq2seq+attention PyTorch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

The code that builds the training corpus is my own code. It takes NLTK's version of the CMU Pronunciation Dictionary and builds a training corpus of over 122,000 training instances.
