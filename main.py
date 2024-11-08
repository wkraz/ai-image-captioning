import torch
import os
import kagglehub
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dotenv import load_dotenv
import pandas as pd
from PIL import Image

load_dotenv()


# Download the Flickr8k dataset
path = kagglehub.dataset_download("adityajn105/flickr8k")
dataset_path = os.getenv('DATASET_PATH')
images_path = images_path = os.path.join(dataset_path, 'Images')
captions_file = os.path.join(dataset_path, 'captions.txt')

# Example: Assuming captions file is in a two-column format (image_id, caption)
captions_df = pd.read_csv(captions_file, sep='\t', names=['image', 'caption'])
captions_dict = captions_df.groupby('image')['caption'].apply(list).to_dict()

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Example function to load and transform an image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform(image)

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
        if self.transform:
            image = self.transform(image)
        caption = self.captions_dict[image_id]
        return image, caption
    
# Instantiate the dataset and dataloader
dataset = Flickr8kDataset(images_path, captions_dict, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# now we can do a training loop
for images, captions in dataloader:
    # Use images and captions in your model
    print("Batch of Images Shape:", images.shape)
    print("Batch of Captions:", captions)
    break  # Remove after verifying output
