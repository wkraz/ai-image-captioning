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
            # Convert batch of captions to tensors and pad them to the same length
            caption_tensors = [caption_to_tensor(caption, vocab) for caption in captions_batch]
            captions = pad_sequence(caption_tensors, batch_first=True, padding_value=vocab["<pad>"])

            # Split captions into inputs and targets
            inputs = captions[:, :-1]  # All tokens except last for input
            targets = captions[:, 1:]  # All tokens except first for target

            # Check if batch sizes match
            if image_features.size(0) != inputs.size(0):
                print("Skipping batch due to size mismatch:", image_features.size(0), inputs.size(0))
                continue

            # Forward pass through decoder
            outputs = decoder(image_features, inputs)  # (batch_size, seq_length, vocab_size)

            # Adjust lengths if necessary to ensure they match
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
