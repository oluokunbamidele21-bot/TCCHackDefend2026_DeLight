from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
CSV_PATH = r'C:\Users\oluok\cassava_AI\data\train.csv'
IMG_DIR = r'C:\Users\oluok\cassava_AI\data\train_images'
df = pd.read_csv(CSV_PATH)
df['label'] = df['label'].astype(str)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=IMG_DIR,
    x_col='image_id',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=IMG_DIR,
    x_col='image_id',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    subset='validation',
    shuffle=False
)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base layers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(5, activation='softmax')  # 5 cassava classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


labels = df['label'].astype(int).values
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels), 
    y=labels
)
class_weights = dict(enumerate(class_weights_array))
print("Class Weights:", class_weights)



# This saves the BEST model automatically during training
checkpoint = ModelCheckpoint(
    'cassava_best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop]
)

model.save(MODEL_SAVE_PATH)
print("Model saved successfully.")

















