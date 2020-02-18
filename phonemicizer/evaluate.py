import random

import torch
import fire
import textdistance


from phonemicizer.data import (
    alphabet,
    phonemes,
    pairs,
)
from phonemicizer.predict import TrainedPhonemicizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lsimilarity(a, b) -> float:
    """
    Returns the Levenshtein similarity between a and b.
    1 means they are equal sequences. 0 means they are
    completely different i.e. their Levenshtein distance
    is maximized.
    """
    return 1 - textdistance.levenshtein.distance(a, b) / max(len(a), len(b))


def evaluateRandomly(n=10):
    model = TrainedPhonemicizer()
    print(
        """
KEY:
    ">" - the input English word
    "=" - the target arpabet representation
    "<" - the model's produced arpabet representation
    """
    )
    similarities = 0.0
    for i in range(n):
        source_indices, target_indices = random.choice(pairs)
        source = alphabet.indices_to_word(source_indices)
        target = phonemes.indices_to_word(target_indices)
        print(">", source)
        print("=", target)
        output_chars, attentions = model.evaluate(source_indices)
        output_word = " ".join(output_chars)
        print("<", output_word)
        similarity = lsimilarity(phonemes.word_to_indices(output_word), target_indices)
        similarities += similarity
        print(f"score = {similarity:.2f}")
        print("")

    print(f"average score = {similarities / n}")


def main(n=10):
    evaluateRandomly(n)


if __name__ == "__main__":
    fire.Fire(main)
