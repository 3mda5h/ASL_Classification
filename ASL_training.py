"""
ASL image classification using neural network of letters A-E, numbers 0-9, and phrases 'bye', 'good', 'good morning', 
'hello', 'little bit', 'no', 'pardon', 'please', 'project', 'whats up', 'yes', and 'NULL' 
(NULL meaning none of the above/no hand in image)


Citation: Mavi. A., and Dikle, Z. (2022). A New 27 Class Sign Language Dataset Collected from 173 Individuals. arXiv:2203.03859
Thanks to all of the volunteer students and teachers from Ayrancı Anadolu High School for their help in collecting data!
Data source: https://www.kaggle.com/datasets/ardamavi/27-class-sign-language-dataset
More info on data: https://arxiv.org/pdf/2203.03859.pdf
I used this tutorial: https://www.tensorflow.org/tutorials/images/classification

Last Updated: 1/26/23
Emily MacPherson
"""  

import matplotlib.pyplot as plt
import PIL

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

images_path = "C:\\Users\\MacAttack\\Downloads\\archive\\X.npy"
labels_path = "C:\\Users\\MacAttack\\Downloads\\archive\\Y.npy"
#load .npy files
images = np.load(images_path)
string_labels = np.load(labels_path)
#for some reason the labels array is a 2d array with a 2nd dimension of 1, change it to an actual 1d array
string_labels = string_labels.flatten()

#array of indexes of all mislabled images I could find
#(there are likely more that I couldn't find)
indexes = [1853, 6835, 10084, 16926, 17451, 17477, 18088, 10911, 13656, 13840, 13885, 14602, 19038,
          20018, 16926, 15009, 10037, 7919, 6849, 6835, 6037, 5471, 4773, 4637, 4617, 4414, 2980,
          2057, 1241, 1208, 1082, 1166, 902, 6999, 6866, 6644, 2980, 2119, 2057, 1532, 1420, 1323,
          914, 910, 861, 21565, 20065, 13623, 13621, 13544, 7243, 7175, 7124, 850, 821]

correct_labels = ['yes', '8', 'please', '6', '6', '6', '2', 'hello', 'a', 'a', 'bye', 'good', 'd',
                  '1', '6', 'e', 'please', '9', 'whats up', 'whats up', '2', '4', '8', '8', '8',
                  '8', '8', 'bye', '8', '8', '8', '8', '8', 'whats up', 'whats up', 'good', '8', 'e',
                  'bye', '7', 'please', 'please', '8', 'please', '8', 'please', '7', 'bye', 'bye', 'a',
                  'good', 'good', 'good', 'please', 'please']

#fix mislabled images *skull emoji* 
for i in range(len(indexes)):
    string_labels[indexes[i]] = correct_labels[i]

print(np.unique(string_labels))

le = preprocessing.LabelEncoder()
#tf wants classes to be numbers not strings, labelEncoder assigns a number to each class
int_labels = le.fit_transform(string_labels)

batch_size = 64
img_height = 128
img_width = 128

#split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(images, int_labels, test_size=0.20, random_state=42)

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

num_classes = 27

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", input_shape = (img_height,img_width, 3)),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
    ])

#not sure how many layers are really ideal, but this seemed to work
#don't need to rescale RGB values, since they came scaled in data  
model = Sequential ([
    data_augmentation,
    layers.Conv2D(16, 3, padding = "same", activation= "relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding = "same", activation= "relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding = "same", activation= "relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding = "same", activation= "relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding = "same", activation= "relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding = "same", activation= "relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding = "same", activation= "relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation = "relu"),
    layers.Dense(num_classes, name = "outputs")
    ])

early_stopping = EarlyStopping(
    min_delta=0.005, # minimium amount of change to count as an improvement
    patience=15, # how many epochs to wait before stopping
    restore_best_weights=True,
)     
model.build(images.shape)

model.compile(optimizer = "adam", loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ["accuracy"])

model.summary()

epochs = 100

history = model.fit(
    x_train,
    y_train,
    validation_data = (x_test, y_test),
    batch_size = batch_size,
    epochs = epochs,
    callbacks=[early_stopping]
)

#convert the model to Tensorflow Lite 
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

#save model
with open("model9.tflite", "wb") as f:
    f.write(tflite_model)

#create plot of validation vs training accuracy
#this doesn't work for some reason :(
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy of model 9')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss of model 9')
plt.show()


