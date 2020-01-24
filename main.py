import torch
from torch import nn

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, path: str):
        # Initialize mapping of language characters to their index
        with open(path, "r") as f:
            alphabet = f.read().split("\n")

        self.letter2index = {"<SOS>": 0, "<EOS>": 1}
        self.index2letter = {0: "<SOS>", 1: "<EOS>"}

        n_letters = 2
        for letter in alphabet:
            if letter not in self.letter2index:
                self.letter2index[letter] = n_letters
                self.index2letter[n_letters] = letter
                n_letters += 1

    def word_to_indices(word: str) -> list:
        letters = word.split(" ")
        return [self.letter2index[letter] for letter in word]


def load_dataset() -> list:
    alphabet = Lang("alphabet.txt")
    phonemes = Lang("arpabet.txt")
    train_data = []
    with open("cmu_pron_dict_dataset.txt", "r") as f:
        for line in f:
            word, phonetic_spelling = line.split("\t")
            # Add this word/phonemes pair to the training datset
            self.train_data.append(
                (
                    alphabet.word_to_indices(word),
                    phonemes.word_to_indices(phonetic_spelling),
                )
            )
    return train_data


def indices_to_tensor(indices: list):
    return torch.tensor(indices, dtype=torch.long).view(-1, 1)


def pair_to_tensors(pair: tuple) -> tuple:
    return (indices_to_tensor(pair[0]), indices_to_tensor(pair[1]))


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


teacher_forcing_ratio = 0.5


def train(
    input_tensor,
    target_tensor,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    max_length=MAX_LENGTH,
):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
