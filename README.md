# AI that captions images

## Project Structure
src - source code file that has all python scripts
    main.py - main driver that loads configurations and runs the training
    dataset.py - loads and transforms data, and creates the DataLoader
    utils.py - functions that build vocabulary, tokenize captions, and convert them to integer arrays
    model.py - creates the Encoder and Decoder classes
    train.py - the training loop for the model

## Getting Started
1. Clone the repository
2. Install dependencies (recommended to be in a virtual environment)
```
pip install -r requirements.txt
```