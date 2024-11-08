import torch
from dataset import get_dataloader
from model import Encoder, Decoder
from train import train_model
from utils import build_vocab, process_captions

def main():
    # Load vocabulary and dataloader
    vocab, captions_dict = build_vocab()
    dataloader = get_dataloader(captions_dict, vocab)

    # Define model parameters
    embed_size = 256
    hidden_size = 512
    vocab_size = len(vocab)

    # Initialize the encoder and decoder
    encoder = Encoder()
    decoder = Decoder(vocab_size, embed_size, hidden_size)

    # Train the model
    train_model(encoder, decoder, dataloader, vocab)

if __name__ == "__main__":
    main()