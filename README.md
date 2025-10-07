## Convolutional Deep Neural Network for Image Classification:

## AIM:

To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY:

The task at hand involves developing a Convolutional Neural Network (CNN) that can accurately classify handwritten digits ranging from 0 to 9. This CNN should be capable of processing scanned images of handwritten digits, even those not included in the standard dataset.

## Neural Network Model:

<img width="906" height="507" alt="Screenshot 2025-09-11 140752" src="https://github.com/user-attachments/assets/aa098f85-777a-4316-a710-9d352087b22d" />

## DESIGN STEPS:

STEP 1:

Preprocess the MNIST dataset by scaling the pixel values to the range [0, 1] and converting labels to one-hot encoded format.

STEP 2:

Build a convolutional neural network (CNN) model with specified architecture using TensorFlow Keras.

STEP 3:

Compile the model with categorical cross-entropy loss function and the Adam optimizer.

STEP 4:

Train the compiled model on the preprocessed training data for 5 epochs with a batch size of 64.

STEP 5:

Evaluate the trained model's performance on the test set by plotting training/validation metrics and generating a confusion matrix and classification report. Additionally, make predictions on sample images to demonstrate model inference.

## PROGRAM:

**Name:** JENISHA TEENA ROSE F

**Register Number:** 2305001010
~~~
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics.head()
print("KAVIPRIYA SP-2305002011")
print("KAVIPRIYA SP-2305002011")
metrics[['accuracy','val_accuracy']].plot()
print("KAVIPRIYA SP-2305002011")
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print("KAVIPRIYA SP-2305002011")
print(confusion_matrix(y_test,x_test_predictions))
print("KAVIPRIYA SP-2305002011")
print(classification_report(y_test,x_test_predictions))
~~~
## OUTPUT:

## Training Data:

<img width="333" height="587" alt="{01593C23-382D-4C0B-B239-54842F6E1CE0}" src="https://github.com/user-attachments/assets/507a0c41-87e4-4658-a3db-6c016eca8a58" />


## Training Loss, Validation Loss Vs Iteration Plot:
<img width="962" height="678" alt="{4F87FFD7-9501-4E2D-AD77-88515F778337}" src="https://github.com/user-attachments/assets/867f1abc-a6fc-4f2f-b82f-e63b441b3b35" />


## Classification Report:

<img width="601" height="199" alt="{F29B699A-75DC-4768-BDD5-A3721A6CB3A9}" src="https://github.com/user-attachments/assets/b4b48a8f-588e-4970-a227-2cf1502ae102" />

## Confusion Matrix:

<img width="698" height="59" alt="{2DBE200E-3686-40E7-BAC9-681AFDDD1B6B}" src="https://github.com/user-attachments/assets/b6b8a690-d3d3-4565-a577-8c28bf9a4f21" />

## New Sample Data Prediction:

<img width="654" height="57" alt="{523E5A7E-2A34-42A9-A55F-E9E7B4E5DA16}" src="https://github.com/user-attachments/assets/78b670a4-59f8-45e5-a2a8-e9b2c903e89d" />


**RESULT**

Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed successfully.
