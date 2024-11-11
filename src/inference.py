import torch
from torchvision import transforms
from PIL import Image
import os
from model import Encoder, Decoder
from utils import load_captions, build_vocab, tokenize_caption

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models
def load_models(encoder_path, decoder_path, vocab_size, embed_size=256, hidden_size=512, device="cpu"):
    encoder = Encoder(embed_size=embed_size).to(device)
    decoder = Decoder(vocab_size, embed_size, hidden_size).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    encoder.eval()
    decoder.eval()
    return encoder, decoder

# Load vocabulary and models
captions_dict = load_captions()
vocab = build_vocab(captions_dict)
encoder_path = "../encoder_final.pth"
decoder_path = "../decoder_final.pth"
encoder, decoder = load_models(encoder_path, decoder_path, len(vocab), device=device)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Greedy decoding for caption generation
def generate_caption(encoder, decoder, image, vocab, max_length=20, device="cpu"):
    inv_vocab = {v: k for k, v in vocab.items()}
    image = transform(image).unsqueeze(0).to(device)
    features = encoder(image)
    
    # Start token
    inputs = torch.tensor([vocab["<bos>"]], dtype=torch.long, device=device).unsqueeze(0)
    caption = []

    with torch.no_grad():
        for _ in range(max_length):
            outputs = decoder(features, inputs)
            _, predicted = outputs[:, -1, :].max(dim=1)
            word_idx = predicted.item()
            if word_idx == vocab["<eos>"]:
                break
            caption.append(inv_vocab.get(word_idx, "<unk>"))
            inputs = torch.cat([inputs, predicted.unsqueeze(0)], dim=1)

    return ' '.join(caption)

# Beam Search decoding
def generate_caption_beam_search(encoder, decoder, image, vocab, max_length=20, beam_size=3, device="cpu"):
    inv_vocab = {v: k for k, v in vocab.items()}
    image = transform(image).unsqueeze(0).to(device)
    features = encoder(image)
    
    sequences = [[[], 0.0, torch.tensor([vocab["<bos>"]], dtype=torch.long, device=device).unsqueeze(0)]]
    
    with torch.no_grad():
        for _ in range(max_length):
            all_candidates = []
            for seq, score, inputs in sequences:
                outputs = decoder(features, inputs)
                scores, indices = torch.topk(outputs[:, -1, :], beam_size)
                
                for i in range(beam_size):
                    candidate = seq + [inv_vocab.get(indices[0][i].item(), "<unk>")]
                    candidate_score = score - scores[0][i].item()
                    candidate_inputs = torch.cat([inputs, indices[0][i].unsqueeze(0).unsqueeze(0)], dim=1)
                    all_candidates.append([candidate, candidate_score, candidate_inputs])
            
            # Sort by score and select top k beams
            ordered = sorted(all_candidates, key=lambda x: x[1])
            sequences = ordered[:beam_size]
            
            # Stop if <eos> is reached
            if any(seq[-1] == "<eos>" for seq, _, _ in sequences):
                break
    
    # Return best sequence
    best_sequence = min(sequences, key=lambda x: x[1])[0]
    return ' '.join(best_sequence).replace("<eos>", "")

# Load and caption an image
def main():
    image_path = "../man_playing_with_dog.jpg"  # Replace with your test image path
    image = Image.open(image_path).convert("RGB")
    
    print("Greedy Decoding Caption:")
    caption = generate_caption(encoder, decoder, image, vocab, device=device)
    print("Generated Caption:", caption)
    
    print("\nBeam Search Decoding Caption:")
    beam_caption = generate_caption_beam_search(encoder, decoder, image, vocab, device=device)
    print("Generated Beam Search Caption:", beam_caption)

if __name__ == "__main__":
    main()
