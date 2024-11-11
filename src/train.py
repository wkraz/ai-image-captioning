import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from utils import tokenize_caption

def caption_to_tensor(caption, vocab):
    tokens = ["<bos>"] + tokenize_caption(caption) + ["<eos>"]
    return torch.tensor([vocab.get(token, vocab["<unk>"]) for token in tokens], dtype=torch.long)

def train_model(encoder, decoder, dataloader, vocab, num_epochs=5, learning_rate=1e-3):
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0

        for images, captions_batch in dataloader:
            # Forward pass through encoder
            image_features = encoder(images)  # (batch_size, feature_dim)

            # Process captions
            caption_tensors = [caption_to_tensor(caption, vocab) for caption in captions_batch]
            captions = pad_sequence(caption_tensors, batch_first=True, padding_value=vocab["<pad>"])

            # Split captions into inputs and targets
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            # Check if batch sizes match
            if image_features.size(0) != inputs.size(0):
                print("Skipping batch due to size mismatch:", image_features.size(0), inputs.size(0))
                continue

            # Forward pass through decoder
            outputs = decoder(image_features, inputs)

            # Adjust lengths if necessary
            min_length = min(outputs.shape[1], targets.shape[1])
            outputs = outputs[:, :min_length, :]
            targets = targets[:, :min_length]

            # Compute loss
            loss = criterion(outputs.reshape(-1, len(vocab)), targets.reshape(-1))

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # Save the model after each epoch
        torch.save(encoder.state_dict(), f'/content/encoder_epoch_{epoch+1}.pth')
        torch.save(decoder.state_dict(), f'/content/decoder_epoch_{epoch+1}.pth')

    # Optionally save the final model state
    torch.save(encoder.state_dict(), '/content/encoder_final.pth')
    torch.save(decoder.state_dict(), '/content/decoder_final.pth')
