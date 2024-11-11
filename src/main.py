import torch
from dataset import get_dataloader
from model import Encoder, Decoder
from train import train_model
from utils import build_vocab, load_captions

def main():
    # Set device to CUDA if available
    print("Starting main function")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load captions and vocabulary
    print("Loading captions and vocabulary...")
    captions_dict = load_captions()
    vocab = build_vocab(captions_dict)
    print("Captions and vocabulary loaded")

    # Create dataloader
    print("Creating dataloader...")
    dataloader = get_dataloader(captions_dict)
    print("Dataloader loaded")

    # Model parameters
    embed_size = 256
    hidden_size = 512
    vocab_size = len(vocab)

    # Initialize encoder and decoder on the specified device
    print("Initializing encoder and decoder...")
    encoder = Encoder().to(device)
    decoder = Decoder(vocab_size, embed_size, hidden_size).to(device)
    print("Encoder and decoder initialized")

    # Train the model
    print("Training model...")
    train_model(encoder, decoder, dataloader, vocab, device=device)
    print("Model trained.")

if __name__ == "__main__":
    main()
