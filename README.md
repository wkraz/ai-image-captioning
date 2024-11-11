# AI that captions images

## Project Structure
- src - source code file that has all python scripts
    - main.py - main driver that loads configurations and runs the training
    - dataset.py - loads and transforms data, and creates the DataLoader
    - utils.py - functions that build vocabulary, tokenize captions, and convert them to integer arrays
    - model.py - creates the Encoder and Decoder classes
    - train.py - the training loop for the model
    - inference.py - the testing model (must be run after training!!)

## Getting Started
1. Clone the repository and enter the directory
2. Install dependencies (recommended to be in a virtual environment)
```
pip install -r requirements.txt
```
3. Add any images you want to your local directory and change this line in `src/inference.py` to its path:
```
image_path = "________"  # Replace with your test image path
```

## Training
This model was trained with the `Flickr8kdataset` (downloadable as `flickr8kdataset.zip`), which is a dataset of 8,000 images of 
common everyday objects with 5 captions each that describe them. \
The actual training was done in Google colab with a cuda gpu (done by just running `python main.py`) \
The training was split into 5 epochs, with each epoch having 2023 images that were split into batches of 10. \
The average loss linearly decreased from epoch to epoch, starting at ~4.4 in epoch 1/5, and ending with an average loss of 2.4204 in epoch 5/5. \
The final encoder and decoder weights were then uploaded to google drive and stored in: `encoder_final.pth` and `decoder_final.pth`. I then downloaded these from Google drive and moved them locally so that they could be used in `src/inference.py`.

## Testing
As stated in the `Getting Started` section, testing is very simple and is just running `src/inference.py` with an image manually uploaded (currently uses `man_playing_with_dog.jpg`). The program takes the image input and outputs a caption, in this case it was:
```
man and a dog in a field
```

## License
This project is licensed under the MIT License - see below for details:
```
MIT License

Copyright (c) 2024 Will Krzastek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```