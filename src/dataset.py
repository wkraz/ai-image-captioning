import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from dotenv import load_dotenv

load_dotenv()

class Flickr8kDataset(Dataset):
    def __init__(self, images_path, captions_dict, transform=None):
        self.images_path = images_path
        self.captions_dict = captions_dict
        self.image_ids = list(captions_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_path, image_id)
        image = Image.open(image_path).convert('RGB')
        
        # transform the image if it was passed when initialized
        if self.transform:
            image = self.transform(image)
            
        caption = self.captions_dict[image_id]
        return image, caption

def get_dataloader(captions_dict, batch_size=4):
    dataset_path = os.getenv('DATASET_PATH')
    images_path = os.path.join(dataset_path, 'Images')

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = Flickr8kDataset(images_path, captions_dict, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader