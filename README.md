Handwriting Classifier
A binary image classification project that distinguishes between two handwriting styles — Galaktioni and Prokle — using a fine-tuned MobileNetV2 model built with TensorFlow/Keras.

Project Structure
FINAL_PROJECT/
├── dataset/
│   ├── training/          # Training images organized by class
│   │   ├── Galaktioni/
│   │   └── Prokle/
│   ├── training_patches/  # Patched/augmented training data
│   ├── check/             # Images to classify (used in project.py)
│   ├── check new/         # Additional check images
│   ├── check2/
│   └── check4/            # Images to classify (used in try.py)
├── venv/                  # Python virtual environment
├── handwriting_model2.keras  # Saved trained model
├── project.py             # Training script + inference on check/
└── try.py                 # Inference-only script using saved model

Requirements

Python 3.8+
TensorFlow 2.x
scikit-learn
NumPy

Install dependencies:
bashpip install tensorflow scikit-learn numpy

Usage
Training the Model
Run project.py to train the model from scratch and evaluate it on the check/ folder:
bashpython project.py
This script will:

Load and augment training images from dataset/training/
Train MobileNetV2 with a frozen base for 5 epochs (feature extraction)
Unfreeze the top 20 layers and fine-tune for another 5 epochs
Save the trained model as handwriting_model2.keras
Run predictions on all .jpg images in dataset/check/

Running Inference on New Images
Use try.py to classify new images using the already-trained model:
bashpython try.py
This script loads handwriting_model2.keras and classifies all .jpg images in dataset/check4/.
Example output:
img_001.jpg                              → Galaktioni  (confidence: 0.92)
img_002.jpg                              → Prokle      (confidence: 0.87)

Model Architecture
ComponentDetailsBase modelMobileNetV2 (pretrained on ImageNet)Input size160 × 160 × 3HeadGlobalAveragePooling2D → Dense(1, sigmoid)LossBinary Cross-EntropyClassesGalaktioni (0), Prokle (1)
Training strategy:

Phase 1 — Base model frozen, only the classification head is trained (LR: 0.001, RMSprop, 5 epochs)
Phase 2 — Top 20 layers of MobileNetV2 unfrozen for fine-tuning (LR: 0.000001, RMSprop, 5 epochs)
Class imbalance is handled automatically via compute_class_weight


Dataset Layout
Training images must be organized into subdirectories named after each class:
dataset/training/
├── Galaktioni/
│   ├── sample1.jpg
│   └── ...
└── Prokle/
    ├── sample1.jpg
    └── ...
An 80/20 train/validation split is applied automatically during training.

Notes

The model uses MobileNetV2's built-in preprocessing (preprocess_input), which scales pixel values to [-1, 1]. Make sure to apply the same preprocessing when running inference.
The confidence score represents the model's predicted probability that an image belongs to class Prokle (class 1). Scores below 0.5 are classified as Galaktioni.
To classify a different folder, update the check_dir path in try.py.
