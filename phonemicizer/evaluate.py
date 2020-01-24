import random

import torch

from phonemicizer.data import (
    SOS_token, EOS_token, alphabet, phonemes, pairs, MAX_LENGTH, indices_to_tensor
)
from phonemicizer.model import Encoder, Decoder, HIDDEN_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', alphabet.indices_to_word(pair[0]))
        print('=', phonemes.indices_to_word(pair[1]))
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def evaluate(encoder, decoder, index_sequence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = indices_to_tensor(index_sequence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(phonemes.index2letter[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


if __name__ == "__main__":
    encoder = Encoder(alphabet.n_letters, HIDDEN_SIZE).to(device)
    encoder.load_state_dict(torch.load("encoder_state.pt"))
    encoder.eval()

    decoder = Decoder(HIDDEN_SIZE, phonemes.n_letters, dropout_p=0.1).to(device)
    decoder.load_state_dict(torch.load("decoder_state.pt"))
    decoder.eval()
    evaluateRandomly(encoder, decoder)
