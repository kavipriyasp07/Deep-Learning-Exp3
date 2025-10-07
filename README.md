#Convolutional Deep Neural Network for Image Classification

**AIM**

To develop a convolutional neural network (CNN) classification model for the given dataset.

**THEORY**

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

**Neural Network Model**

Include the neural network model diagram.

**DESIGN STEPS**
STEP 1:
Preprocess the MNIST dataset by scaling the pixel values to the range [0, 1] and convertir labels to one-hot encoded format.
STEP 2:
Build a convolutional neural network (CNN) model with specified architecture using TensorFlow Keras.
STEP 3:
Compile the model with categorical cross-entropy loss function and the Adam optimizer.
STEP 4:
Train the compiled model on the preprocessed training data for 5 epochs with a batch siz of 64.
STEP 5:
Evaluate the trained model's performance on the test set by plotting training/validation metrics and generating a confusion matrix and classification report. Additionally, make predictions on sample images to demonstrate model inference.

**PROGRAM**

NAME:KAVIPRIYA SP
REGISTER NUMBER:2305002011
```
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
```


**OUTPUT**

**Training Loss per Epoch**
<img width="438" height="568" alt="{57D0FB57-69B3-4377-830E-3FD6DECD2F13}" src="https://github.com/user-attachments/assets/7a73ca7e-f9d4-4a5e-9dc7-352f7d1b36f2" />


**Confusion Matrix**
<img width="743" height="321" alt="{0AB5F24E-43D4-40FC-A72F-AC427C2A1E89}" src="https://github.com/user-attachments/assets/73db4b33-b550-42f3-89f0-c7a51f135adc" />


**Classification Report**

<img width="717" height="458" alt="{449D3BDD-7D8A-4101-81C7-A0615DCA803C}" src="https://github.com/user-attachments/assets/fabfcb5f-05bd-4857-847f-4266b32055aa" />


**New Sample Data Prediction**
<img width="581" height="491" alt="{894394AD-D4EC-46D8-AB31-9EB09D303585}" src="https://github.com/user-attachments/assets/40c5c703-cb84-465b-9986-2ab2fc192010" />


**RESULT**

Thus, a convolutional deep neural network for digit classification and to verify the response
for scanned handwritten images is developed successfully.
