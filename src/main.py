import torch
from dataset import get_dataloader
from model import Encoder, Decoder
from train import train_model
from utils import build_vocab, load_captions

def main():
    # Load captions and vocabulary
    captions_dict = load_captions()
    vocab = build_vocab(captions_dict)

    # Create dataloader
    dataloader = get_dataloader(captions_dict)

    # Model parameters
    embed_size = 256
    hidden_size = 512
    vocab_size = len(vocab)

    # Initialize encoder and decoder
    encoder = Encoder()
    decoder = Decoder(vocab_size, embed_size, hidden_size)

    # Train the model
    train_model(encoder, decoder, dataloader, vocab)

if __name__ == "__main__":
    main()
