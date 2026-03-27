import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = 160
BATCH_SIZE = 32
keras = tf.keras

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=15,          
    zoom_range=0.2,             
    width_shift_range=0.2,      
    height_shift_range=0.2,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2,
)

train_dir = os.path.join("dataset", "training")

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
)

val_generator = val_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

classes = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(classes),
    y=classes
)
class_weights = dict(enumerate(class_weights))

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,      
    weights='imagenet'  
)

model = keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1, activation='sigmoid')
])

base_model.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator,
    class_weight=class_weights
)

base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.000001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)

history_phase2 = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator,
    class_weight=class_weights
)

acc = history.history['accuracy'] + history_phase2.history['accuracy']
print(acc)

model.save("handwriting_model2.keras")
print("Model Saved!")


check_dir = os.path.join("dataset", "check")
image_paths = glob.glob(os.path.join(check_dir, "*.jpg"))
image_paths.sort()

if len(image_paths) == 0:
    print("No Images!")
else:
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

    class_names = {v: k for k, v in train_generator.class_indices.items()}

    results = [
        (os.path.basename(fpath), class_names[pred], f"{prob:.4f}")
        for fpath, pred, prob in zip(image_paths, predicted_classes, probabilities)
    ]

    for filename, label, prob in results:
        print(f"{filename:40s} → {label}  (confidence: {prob})")

    results_dict = {
        fname: {"class": label, "confidence": prob}
        for fname, label, prob in results
    }