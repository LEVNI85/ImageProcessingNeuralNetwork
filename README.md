# Handwriting Classifier

## Project Structure
- `data/`: Contains datasets.
- `notebooks/`: Jupyter notebooks for experimentation.
- `src/`: Source code for the model and preprocessing.
- `requirements.txt`: Required Python packages.
- `README.md`: Project documentation.

## Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib

Install the requirements using:
```
pip install -r requirements.txt
```

## Usage Instructions
### Training
1. Prepare your dataset in the `data/` folder.
2. Run the training script:
```
python src/train.py
```
3. The model will be saved in the `models/` folder.

### Inference
To make predictions using the trained model:
1. Ensure you have your model stored in the specified location.
2. Use the inference script:
```
python src/inference.py --model_path ./models/model.h5 --input_image ./data/test_image.png
```

## Model Architecture
The Handwriting Classifier uses a Convolutional Neural Network (CNN) architecture with the following layers:
- Input layer: 28x28 grayscale images
- Convolutional layers: Extract features from images
- Fully connected layers: Classify the extracted features
- Output layer: Softmax activation for multiple classes

## Dataset Layout
- `data/train/`: Training images organized into subfolders for each class.
- `data/test/`: Test images organized similarly.

## Notes about Preprocessing
- Images are resized to 28x28 pixels.
- Normalization is applied to pixel values.

## Confidence Scores
- The model outputs confidence scores for each class, indicating the prediction's reliability.

This project aims to provide a robust method for handwritten digit classification using deep learning techniques.