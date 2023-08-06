import pandas as pd
import glob
import csv
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

path = r'/workspaces/correlation/dataset/Preprocessed/Dataset1'
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=None)
    df["Class"] = filename
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
print(frame.shape)

with open('dataset/FlowStats_column_list_with_class.csv', encoding='cp1252') as f:
    reader = csv.reader(f, delimiter=',')
    rows = list(reader)

cols = []
for sublist in rows:
    for item in sublist:
        cols.append(item)

frame.columns = cols

label = {k: v for v, k in enumerate(all_files)}
frame.replace({"Class": label})
print(frame.shape)
data = frame.values
X, y = data[:, : -1], data[:, -1]
print(X.shape, y.shape)

X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=0.3, random_state=25)
X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.5)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
# y_train = to_categorical(y_train)
print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

n_classes = 6
dropout_rate = 0.3
learning_rate = 0.0001

# Building a 1D CNN model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=4, strides=2, padding='same',
                                 activation='relu', input_shape=(178, 1)))
model.add(tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=4, strides=2,
                                 padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same'))
# model.add(BatchNormalization())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(dropout_rate))
model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

print(X_train.shape, y_train.shape)
X_train = np.asarray(X_train)
Y_train = np.asarray(y_train)
print(X_train.shape, y_train.shape)

hist = model.fit(X_train, y_train, batch_size=50, epochs=20)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

pred = model.predict(X_test)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(classification_report(y_test.argmax(axis=1), pred.argmax(axis=1)))

confMat = confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))
cm = ConfusionMatrixDisplay(confusion_matrix=confMat)
cm.plot(xticks_rotation='vertical', cmap='Blues')
cm.ax_.set(xlabel='Predicted', ylabel='True', )