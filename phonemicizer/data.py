import math
import random

import torch


SOS_token = 0
EOS_token = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Lang:
    def __init__(self, path: str):
        # Initialize mapping of language characters to their index
        with open(path, "r") as f:
            alphabet = f.read().split("\n")

        self.letter2index = {"<SOS>": 0, "<EOS>": 1}
        self.index2letter = {0: "<SOS>", 1: "<EOS>"}

        self.n_letters = 2
        for letter in alphabet:
            if letter not in self.letter2index:
                self.letter2index[letter] = self.n_letters
                self.index2letter[self.n_letters] = letter
                self.n_letters += 1

    def word_to_indices(self, word: str) -> list:
        """
        Word will be a spaced word e.g. 'd o g'
        """
        letters = word.split(" ")
        return [self.letter2index[letter] for letter in letters]

    def indices_to_word(self, indices: list) -> str:
        return " ".join(self.index2letter[i] for i in indices)


alphabet = Lang("alphabet.txt")
phonemes = Lang("arpabet.txt")


def load_dataset() -> list:
    data = []
    max_input_seq_length = 0

    with open("cmu_pron_dict_dataset.txt", "r") as f:
        for line in f:
            word, phonetic_spelling = line.replace("\n", "").split("\t")
            # Add this word/phonemes pair to the training datset
            alphabet_word_indices = alphabet.word_to_indices(word)
            if len(alphabet_word_indices) > max_input_seq_length:
                max_input_seq_length = len(alphabet_word_indices)
            phons_word_indices = phonemes.word_to_indices(phonetic_spelling)
            data.append((alphabet_word_indices, phons_word_indices,))

    # Make an 80/20 train/validation split.
    n = len(data)
    n_train = math.floor(n * 0.8)
    train_data = data[:n_train]
    val_data = data[n_train:]
    return train_data, val_data, max_input_seq_length


train_pairs, val_pairs, MAX_LENGTH = load_dataset()


def indices_to_tensor(indices: list):
    return torch.tensor(indices, dtype=torch.long, device=device).view(-1, 1)


def pair_to_tensors(pair: tuple) -> tuple:
    return (indices_to_tensor(pair[0]), indices_to_tensor(pair[1]))
