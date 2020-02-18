import time
import math
import random

import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import fire

from phonemicizer.data import (
    SOS_token,
    EOS_token,
    alphabet,
    phonemes,
    pairs,
    MAX_LENGTH,
    pair_to_tensors,
)
from phonemicizer.model import Encoder, Decoder, HIDDEN_SIZE

# NOTE: The code in this file follows this pytorch tutorial closely:
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# That tutorial does translation from English word sequences to French word
# sequences. This model does translation from English character sequences
# to phonetic character sequences.


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))


def train(
    input_tensor,
    target_tensor,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    max_length,
    teacher_forcing_ratio,
):
    """
    Train a single data instance
    """
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

    return loss.item() / target_length


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig("loss-curve.png")


def train_iters(
    encoder,
    decoder,
    n_iters: int,
    *,
    max_length: int,
    teacher_forcing_ratio: float,
    print_every: int,
    plot_every: int,
    learning_rate: float,
    weight_decay: float,
):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(
        encoder.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    decoder_optimizer = optim.Adam(
        decoder.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    training_pairs = [pair_to_tensors(random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(
            input_tensor,
            target_tensor,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
            max_length,
            teacher_forcing_ratio,
        )
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(
                "%s (%d %d%%) %.4f"
                % (
                    timeSince(start, iter / n_iters),
                    iter,
                    iter / n_iters * 100,
                    print_loss_avg,
                )
            )

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


def main(
    n_iters,
    print_every: int = 1000,
    plot_every: int = 100,
    learning_rate: float = 0.0001,
    # L2 weight decay
    weight_decay: float = 0.0,
    teacher_forcing_ratio: float = 0.5,
    dropout_p: float = 0.1,
):
    print(f"Training for {n_iters} iterations...")
    encoder = Encoder(alphabet.n_letters, HIDDEN_SIZE).to(device)
    decoder = Decoder(HIDDEN_SIZE, phonemes.n_letters, dropout_p=dropout_p).to(device)
    train_iters(
        encoder,
        decoder,
        n_iters,
        print_every=print_every,
        plot_every=plot_every,
        learning_rate=learning_rate,
        max_length=MAX_LENGTH,
        teacher_forcing_ratio=teacher_forcing_ratio,
        weight_decay=weight_decay,
    )

    print("Training finished. Saving models...")

    torch.save(encoder.state_dict(), "encoder_state.pt")
    torch.save(decoder.state_dict(), "decoder_state.pt")


if __name__ == "__main__":
    fire.Fire(main)
