# Import required libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

base_dir = './data/casting_dataset/'
train_dir = os.path.join(base_dir, 'train/')
test_dir = os.path.join(base_dir, 'test/')

# Print class counts
def print_class_counts(directory, class_names):
    for class_name in class_names:
        path = os.path.join(directory, class_name)
        print(f'{class_name}: {len(os.listdir(path))} images')

print('Training set:')
print_class_counts(train_dir, ['ok_front', 'def_front'])
print('\nTest set:')
print_class_counts(test_dir, ['ok_front', 'def_front'])

# Data generators
img_size = (300, 300)
batch_size = 64

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode='grayscale',
    class_mode='binary',
    batch_size=batch_size,
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    color_mode='grayscale',
    class_mode='binary',
    batch_size=batch_size,
    subset='validation')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    color_mode='grayscale',
    class_mode='binary',
    batch_size=batch_size,
    shuffle=False)

# Build CNN model
model = Sequential([
    Conv2D(32, 3, activation='relu', padding='same', strides=2, input_shape=(300, 300, 1)),
    MaxPooling2D(2, strides=2),
    Conv2D(64, 3, activation='relu', padding='same', strides=2),
    MaxPooling2D(2, strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    verbose=1)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_generator)
print(f'\nTest accuracy: {test_acc:.4f}')

# Predictions
y_pred = (model.predict(test_generator) >= 0.5).astype(int)
y_true = test_generator.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['OK', 'Defective'], 
            yticklabels=['OK', 'Defective'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print('\nClassification Report:')
print(classification_report(y_true, y_pred, target_names=['OK', 'Defective']))
