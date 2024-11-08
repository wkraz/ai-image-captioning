import torch.nn as nn
import torchvision.models as models
import torch

class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__() # calling init method of superclass (nn.Module) - sets it up as a neural network
        resnet = models.resnet50(pretrained=True) # pretrained resnet50 model
        self.encoder = nn.Sequential(*list(resnet.children())[:-2]) # converts resnet50 model into nn.sequential module
        self.pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size)) # pooling layer that takes adaptive size -> fixed
    
    def forward(self, images):
        features = self.encoder(images)
        features = self.pool(features)
        return features

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size) # maps each word in vocabulary -> vector representation
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True) # necessary for long term learning
        self.linear = nn.Linear(hidden_size, vocab_size) # linear layer that acts as a classifier
    
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs
