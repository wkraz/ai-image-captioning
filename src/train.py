import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def caption_to_tensor(caption, vocab):
    # Tokenize the caption and add <bos> and <eos> tokens
    tokens = ["<bos>"] + caption.lower().split() + ["<eos>"]
    return torch.tensor([vocab[token] for token in tokens], dtype=torch.long)

def train_model(encoder, decoder, dataloader, vocab, num_epochs=5, learning_rate=1e-3):
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    for epoch in range(num_epochs):
        for images, captions_batch in dataloader:
            # Forward pass through encoder
            image_features = encoder(images)
            
            # Process captions
            captions = pad_sequence([caption_to_tensor(caption, vocab) for caption in captions_batch],
                                    batch_first=True, padding_value=vocab["<pad>"])
            inputs = captions[:, :-1] # inputs exclude the last token
            targets = captions[:, 1:] # targets exclude the first token

            # Forward pass through decoder
            outputs = decoder(image_features, inputs)
            loss = criterion(outputs.view(-1, len(vocab)), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")