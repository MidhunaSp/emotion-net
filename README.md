## Overview

The notebook trains the full-size emotion CNN (~3.5M params) on the
FER2013 dataset using a free Colab GPU (T4). It's the GPU counterpart
to the lightweight CPU version used for local/sandbox training — same
core pipeline (class-weighted loss, augmentation, early stopping) but
with a bigger model and more epochs since GPU compute allows it.

## Environment

- **Platform:** Google Colab, T4 GPU runtime
- **Framework:** TensorFlow / Keras
- **Dataset:** FER2013, folder-of-images format (`train/<emotion>/*.jpg`,
  `test/<emotion>/*.jpg`), uploaded as a zip directly into the notebook

## Step-by-step

### 1. Upload dataset
The dataset zip is uploaded interactively via Colab's file picker
(`google.colab.files.upload()`) and extracted.

### 2. Locate train/test folders
The notebook auto-detects the `train/` and `test/` directories inside
the extracted zip, regardless of nesting, by walking the directory tree.

### 3. Load images into arrays
Each image is:
- opened and converted to grayscale
- resized to 48x48
- normalized to [0, 1]

Labels are assigned from the 7 folder names (angry, disgust, fear,
happy, sad, surprise, neutral), matching FER2013's standard class
ordering. A 10% stratified split is carved out of the training folder
for validation, since this folder layout only ships train/test.

### 4. Compute class weights
`sklearn.utils.class_weight.compute_class_weight(class_weight="balanced")`
is used to counter FER2013's severe class imbalance — Happy has
roughly 16-17x more training images than Disgust. Each class gets a
weight inversely proportional to its frequency, which the loss
function uses to penalize mistakes on minority classes more heavily.

### 5. Model architecture
A VGG-style CNN, `build_emotion_cnn()`:

| Block | Layers | Filters |
|---|---|---|
| 1 | 2x Conv2D + BatchNorm + ReLU, MaxPool, Dropout(0.25) | 64 |
| 2 | 2x Conv2D + BatchNorm + ReLU, MaxPool, Dropout(0.25) | 128 |
| 3 | 2x Conv2D + BatchNorm + ReLU, MaxPool, Dropout(0.3) | 256 |
| Head | Dense(256) + BatchNorm + ReLU, Dropout(0.5), Dense(7, softmax) | — |

L2 regularization (1e-4) is applied to all conv/dense layers.
~3.5M total parameters.

### 6. Training configuration
- **Optimizer:** Adam, initial learning rate 1e-3
- **Loss:** sparse categorical crossentropy
- **Batch size:** 64
- **Max epochs:** 60 (early stopping usually ends it sooner)
- **Augmentation** (`ImageDataGenerator`): rotation ±10°, width/height
  shift 10%, zoom 10%, horizontal flip — mild augmentation since
  FER2013 faces are already tightly cropped
- **Callbacks:**
  - `EarlyStopping` — stops if validation loss doesn't improve for 8
    epochs, restores best weights
  - `ReduceLROnPlateau` — halves the learning rate if validation loss
    plateaus for 3 epochs (down to a floor of 1e-6)
  - `ModelCheckpoint` — saves the model only when validation accuracy
    improves

### 7. Evaluation
After training, the model is evaluated once on the held-out test set
(images never seen during training or validation). Reports:
- Overall test accuracy and loss
- Per-class precision/recall/F1 (`sklearn.metrics.classification_report`)
- Confusion matrix, to see which emotions get confused with each other
  (Fear/Sad/Neutral tend to overlap most)

### 8. Training curves
Matplotlib plots of train/val accuracy and train/val loss per epoch,
saved as `training_curves.png`.

### 9. Download
The best checkpoint (`emotion_model.h5`) and the training curve image
are downloaded to your local machine via Colab's `files.download()`.

## Reproducing this run

1. Open `train_on_colab.ipynb` in Google Colab
2. Runtime → Change runtime type → GPU (T4)
3. Runtime → Run all
4. Upload your `archive.zip` (FER2013, folder format) when prompted
5. Wait for training to finish (early stopping typically triggers
   somewhere before the 60-epoch cap)
6. Download `emotion_model.h5` and `training_curves.png` at the end
7. Drop `emotion_model.h5` into your local `models/` folder to use it
   with `src/realtime_app.py`
