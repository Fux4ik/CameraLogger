import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Function to load images
def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    return images

X_train = []
y_train = []

# Person = class 1
for img in load_images('dataset/train/person'):
    X_train.append(img)
    y_train.append(1)

# No person = class 0
for img in load_images('dataset/train/no_person'):
    X_train.append(img)
    y_train.append(0)

# Loading test data
X_test = []
y_test = []

for img in load_images('dataset/test/person'):
    X_test.append(img)
    y_test.append(1)

for img in load_images('dataset/test/no_person'):
    X_test.append(img)
    y_test.append(0)

# Converting to numpy arrays + normalizing
X_train = np.array(X_train).reshape(-1, 200, 250, 1) / 255.0
X_test = np.array(X_test).reshape(-1, 200, 250, 1) / 255.0
y_train = np.array(y_train)
y_test = np.array(y_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Simple model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(200,250,1)),
    MaxPool2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary
model.summary()

# Train
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Evaluating
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, predicted_classes))

# Save
model.save('person_detector.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('person_detector.tflite', 'wb') as f:
    f.write(tflite_model)