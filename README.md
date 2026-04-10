# Image Processing Neural Network

A binary image classifier built with TensorFlow and MobileNetV2 transfer learning. The model is trained to distinguish between two handwriting styles — **Galaktioni** and **Prokle** — using a two-phase fine-tuning approach.

---

## Overview

This project trains a convolutional neural network on a custom image dataset using MobileNetV2 as a pretrained backbone. It uses a two-phase training strategy: first training only the classification head, then fine-tuning the top layers of the base model for improved accuracy.

---

## Project Structure

```
ImageProcessingNeuralNetwork/
├── project.py          # Full training pipeline + inference on new images
├── try.py              # Standalone inference script using a saved model
├── dataset/
│   ├── training/       # Training images, organized into subdirectories per class
│   │   ├── Galaktioni/
│   │   └── Prokle/
│   └── check/          # Images to run inference on after training
```

> **Note:** The `dataset/` directory and trained model files (`.keras`) are excluded from version control via `.gitignore`.

---

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- scikit-learn

Install dependencies:

```bash
pip install tensorflow numpy scikit-learn
```

---

## Dataset Setup

Organize your training images into subdirectories named after each class inside `dataset/training/`:

```
dataset/training/
├── Galaktioni/
│   ├── image1.jpg
│   └── ...
└── Prokle/
    ├── image1.jpg
    └── ...
```

Place any images you want to classify after training in `dataset/check/` as `.jpg` files.

---

## Training

Run the full training pipeline:

```bash
python project.py
```

**What it does:**

1. **Data augmentation** — Applies rotation, zoom, and shifts to the training images to improve generalization.
2. **Phase 1 (5 epochs)** — Freezes the MobileNetV2 base and trains only the classification head (`GlobalAveragePooling2D` + `Dense(1, sigmoid)`) using RMSprop at `lr=0.001`.
3. **Phase 2 (5 epochs)** — Unfreezes the last 20 layers of MobileNetV2 and fine-tunes with a much lower learning rate (`lr=0.000001`).
4. **Class balancing** — Automatically computes class weights to handle imbalanced datasets.
5. **Inference** — After training, runs predictions on all `.jpg` images in `dataset/check/` and prints the predicted class with confidence score.
6. **Model saved** to `handwriting_model2.keras`.

---

## Inference on New Images

To run inference using a previously saved model without retraining:

```bash
python try.py
```

This loads `handwriting_model2.keras` and classifies all `.jpg` images in `dataset/check4/`, printing results in the format:

```
image_001.jpg                            → Galaktioni  (confidence: 0.97)
image_002.jpg                            → Prokle      (confidence: 0.83)
```

---

## Model Architecture

| Layer                  | Details                          |
|------------------------|----------------------------------|
| Base model             | MobileNetV2 (pretrained ImageNet, no top) |
| Input size             | 160 × 160 × 3                    |
| Pooling                | GlobalAveragePooling2D           |
| Output                 | Dense(1, activation=`sigmoid`)   |
| Loss                   | Binary cross-entropy             |
| Phase 1 optimizer      | RMSprop, lr=0.001                |
| Phase 2 optimizer      | RMSprop, lr=0.000001             |

---

## Configuration

Key constants at the top of `project.py`:

| Constant     | Default | Description                  |
|--------------|---------|------------------------------|
| `IMG_SIZE`   | `160`   | Input image size (px)        |
| `BATCH_SIZE` | `32`    | Training batch size          |

---

## Output

- **`handwriting_model2.keras`** — Saved trained model.
- Console output of per-image predictions and confidence scores after training.
