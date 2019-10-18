from load_data import Dataset

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from create_model import create_model
data = Dataset()


data_train, data_test, labels_train, labels_test = train_test_split(
    data.dataset, data.labels, test_size=0.20, random_state=42)

class_names = ["Not fire", "Fire"]

plt.figure()
plt.imshow(data_train[1])
plt.colorbar()
plt.grid(False)
plt.show()
data_train = data_train / 255
data_test = data_test / 255
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(data_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels_train[i]])
plt.show()

model = create_model()
print(model.summary())

model.save_weights('./checkpoints/my_checkpoint')

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data_train, labels_train, epochs=20)

test_loss, test_acc = model.evaluate(data_test, labels_test, verbose=2)

print('\nTest accuracy:', test_acc)
print('\nTest loss: ', test_loss)