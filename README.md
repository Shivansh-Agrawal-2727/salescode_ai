# Real Image vs Screen Image Detection

This project trains a machine-learning model to classify an image as either a real-world photo or a photo of a screen. It uses the dataset layout already present in this folder:

```text
dataset/
  real/
  screen/
```

## What The Project Does

- Reads JPG, PNG, WebP, TIFF, HEIC, and HEIF images.
- Extracts image-processing features such as color statistics, saturation, contrast, edge density, sharpness, and high-frequency screen patterns.
- Trains a Random Forest classifier.
- Saves the trained model.
- Creates an evaluation report with accuracy, classification report, confusion matrix, and cross-validation accuracy.
- Predicts new images as `real` or `screen`.

## Setup

Create or activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Train The Model

Run:

```bash
python train.py
```

Outputs:

```text
models/screen_real_classifier.joblib
reports/evaluation.json
```

## Predict An Image

For one image:

```bash
python predict.py path/to/image.jpg
```

For a folder:

```bash
python predict.py path/to/folder
```

Each result includes the predicted class, confidence, and probabilities for both classes.

## Method Used

The classifier uses handcrafted image-processing features instead of a heavy deep-learning model. This is a good fit for a small assignment dataset because it is fast, explainable, and works with limited training images.

Important features include RGB statistics, HSV statistics, contrast, edge density, Laplacian sharpness, FFT high-frequency energy, line energy, entropy, and exposure ratios.

## Project Structure

```text
screen_real_detector/
  features.py   feature extraction
  io.py         image loading and HEIC support
  model.py      dataset loading, training, saving, prediction
train.py        trains and evaluates the classifier
predict.py      classifies new image files or folders
requirements.txt
dataset/
```
