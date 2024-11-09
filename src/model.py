import torch.nn as nn
import torchvision.models as models
import torch

class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14, embed_size=256):
        super(Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # Add a linear layer to reduce the dimension to `embed_size`
        self.fc = nn.Linear(401408, embed_size)  # Adjust based on the flattened output size of your ResNet encoder

    def forward(self, images):
        features = self.encoder(images)
        features = self.pool(features)
        features = features.view(features.size(0), -1)  # Flatten to (batch_size, feature_dim)
        features = self.fc(features)  # Map to (batch_size, embed_size)
        return features


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # Expand features to 3D: (batch_size, 1, feature_dim)
        features = features.unsqueeze(1)

        # Pass captions through embedding layer
        embeddings = self.embed(captions)  # (batch_size, seq_length, embed_size)

        # Concatenate features and embeddings
        inputs = torch.cat((features, embeddings), dim=1)  # (batch_size, seq_length+1, embed_size)

        # LSTM forward pass
        outputs, _ = self.lstm(inputs)  # (batch_size, seq_length+1, hidden_size)

        # Map outputs to vocab size
        outputs = self.linear(outputs)  # (batch_size, seq_length+1, vocab_size)

        return outputs


