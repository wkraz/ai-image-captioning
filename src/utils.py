import csv
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
import os
from dotenv import load_dotenv

load_dotenv()

tokenizer = TreebankWordTokenizer()

# Tokenize a single caption
def tokenize_caption(caption):
    return tokenizer.tokenize(caption.lower())


# Build vocabulary from the tokenized captions
def build_vocab(captions_dict, min_freq=1):
    counter = Counter()
    for captions in captions_dict.values():
        for caption in captions:
            tokens = tokenize_caption(caption)
            counter.update(tokens)

    # Assign each word in the vocabulary an index
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

# Load and process captions
def load_captions():
    dataset_path = os.getenv('DATASET_PATH')
    captions_file = os.path.join(dataset_path, 'captions.txt')
    
    with open(captions_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        captions_data = [(row[0].strip(), row[1].strip()) for row in reader]
    
    captions_df = pd.DataFrame(captions_data, columns=['image', 'caption'])
    captions_dict = captions_df.groupby('image')['caption'].apply(list).to_dict()
    
    return captions_dict
