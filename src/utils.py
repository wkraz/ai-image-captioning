import csv
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator

def tokenize_caption(caption):
    return caption.lower().split()

def yield_tokens(captions_dict):
    for captions in captions_dict.values():
        for caption in captions:
            yield tokenize_caption(caption)

def build_vocab():
    dataset_path = os.getenv('DATASET_PATH')
    captions_file = os.path.join(dataset_path, 'captions.txt')
    
    # Load and process captions
    with open(captions_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        captions_data = [(row[0].strip(), row[1].strip()) for row in reader]
    
    captions_df = pd.DataFrame(captions_data, columns=['image', 'caption'])
    captions_dict = captions_df.groupby('image')['caption'].apply(list).to_dict()
    
    vocab = build_vocab_from_iterator(yield_tokens(captions_dict), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
    vocab.set_default_index(vocab["<unk>"])
    
    return vocab, captions_dict
