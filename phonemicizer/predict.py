import re

import torch
from nltk.tokenize import TweetTokenizer
import fire

from phonemicizer.data import (
    SOS_token,
    EOS_token,
    alphabet,
    phonemes,
    MAX_LENGTH,
    indices_to_tensor,
)
from phonemicizer.model import Encoder, Decoder, HIDDEN_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainedPhonemicizer:
    """
    Uses a trained phonemicizer to translate english
    strings to their phonetic representations.
    """

    def __init__(self):
        self.tokenizer = TweetTokenizer()
        self.supported_characters = set(alphabet.letter2index.keys())
        self.supported_characters.add(" ")
        self.blacklisted_output_characters = {
            phonemes.index2letter[EOS_token],
            phonemes.index2letter[SOS_token],
        }

        self.encoder = Encoder(alphabet.n_letters, HIDDEN_SIZE).to(device)
        self.encoder.load_state_dict(torch.load("encoder_state.pt"))
        self.encoder.eval()

        self.decoder = Decoder(HIDDEN_SIZE, phonemes.n_letters, dropout_p=0.0).to(
            device
        )
        self.decoder.load_state_dict(torch.load("decoder_state.pt"))
        self.decoder.eval()

    def predict(self, input_str: str):
        """
        `input_str` can be any string, not just a word. Returns the
        predicted phonetic representation of `input_str`.
        """
        input_str = input_str.lower()
        scrubbed_input_str = "".join(
            [c for c in input_str if c in self.supported_characters]
        )
        input_words = self.tokenizer.tokenize(scrubbed_input_str)
        input_words = [word for word in input_words if re.search("[a-zA-Z]", word)]
        output_words = []
        for word in input_words:
            indices = alphabet.word_to_indices(" ".join(word))
            indices = [SOS_token] + indices + [EOS_token]
            output_chars, attentions = self.evaluate(indices)
            output_chars = [
                c for c in output_chars if c not in self.blacklisted_output_characters
            ]
            output_word = " ".join(output_chars)
            output_words.append(output_word)
        return " . ".join(output_words)

    def evaluate(self, index_sequence):
        """
        A lower level version of `self.predict`. Accepts an already cleaned
        indecized version of the input, and only accepts a single word at a
        time.
        """
        with torch.no_grad():
            input_tensor = indices_to_tensor(index_sequence)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden()

            encoder_outputs = torch.zeros(
                MAX_LENGTH, self.encoder.hidden_size, device=device
            )

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(
                    input_tensor[ei], encoder_hidden
                )
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH)

            for di in range(MAX_LENGTH):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append("<EOS>")
                    break
                else:
                    decoded_words.append(phonemes.index2letter[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[: di + 1]


def main(i: str):
    model = TrainedPhonemicizer()
    print(model.predict(i))


if __name__ == "__main__":
    fire.Fire(main)
