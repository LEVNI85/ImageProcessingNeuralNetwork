import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

IMG_SIZE = 160

model = tf.keras.models.load_model("handwriting_model2.keras")

check_dir = os.path.join("dataset", "check4")
image_paths = glob.glob(os.path.join(check_dir, "*.jpg"))
image_paths.sort()


images = []
for img_path in image_paths:
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    images.append(img_array)

images = np.array(images)
images = tf.keras.applications.mobilenet_v2.preprocess_input(images)

raw_preds = model.predict(images)
probabilities = raw_preds.flatten()
predicted_classes = (probabilities >= 0.5).astype(int)

CLASS_NAMES = {0: "Galaktioni", 1: "Prokle"}

for fpath, pred, prob in zip(image_paths, predicted_classes, probabilities):
    label = CLASS_NAMES[pred]
    print(f"{os.path.basename(fpath):40s} → {label}  (confidence: {prob:.2f})")