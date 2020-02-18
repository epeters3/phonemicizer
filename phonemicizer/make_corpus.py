import re

import nltk

entries = nltk.corpus.cmudict.entries()


def clean_phoneme(pron: str) -> str:
    return re.sub("[0-9]", "", pron)


def build_pron_entry(pron: list) -> str:
    return " ".join([clean_phoneme(phon) for phon in pron])


alphabet = set()
phonemes = set()

with open("cmu_pron_dict_dataset.txt", "w") as dataset_f:
    for word, pron in entries:
        for letter in word:
            alphabet.add(letter)
        for phon in pron:
            phonemes.add(clean_phoneme(phon))
        dataset_f.write(
            f"<SOS> {' '.join(word)} <EOS>\t<SOS> {build_pron_entry(pron)} <EOS>\n"
        )

with open("arpabet.txt", "w") as phonemes_f:
    # Write a list of all unique phonemes
    phonemes_f.write("\n".join(phonemes))

with open("alphabet.txt", "w") as alphabet_f:
    # Write a list of all unique letters
    alphabet_f.write("\n".join(alphabet))
