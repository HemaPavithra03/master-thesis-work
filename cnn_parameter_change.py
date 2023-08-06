import pandas as pd
import glob
import csv
import sklearn.model_selection
import train_test_split

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
print(label)
frame.replace({"Class": label})
fr = frame.values
X, y = fr[:, :-1], fr[:, -1]
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=25, stratify=y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print("Build model")
model_1D_CNN = tf.keras.models.Sequential()
model_1D_CNN.add(tf.keras.layers.Conv1D(32, 25, strides=1, padding='same', a
ctivation = 'relu', input_shape = (784, 1)))
model_1D_CNN.add(tf.keras.layers.MaxPooling1D(pool_size=3, strides=3, paddin
g = 'same'))
model_1D_CNN.add(tf.keras.layers.Conv1D(64, 25, strides=1, padding='same', a
ctivation = 'relu'))
model_1D_CNN.add(tf.keras.layers.MaxPooling1D(pool_size=3, strides=3, paddin
g = 'same'))
model_1D_CNN.add(tf.keras.layers.Flatten())
model_1D_CNN.add(tf.keras.layers.Dense(1024, activation='relu'))
model_1D_CNN.add(tf.keras.layers.Dropout(0.5))
model_1D_CNN.add(tf.keras.layers.Dense(numClasses, activation='softmax'))

model_1D_CNN.summary()

print("Training")
# Parameters
learningRate = 1e-4
batchSize = 50
numEpoch = 50
# Initialization
optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)  # t
f.keras.optimizers.SGD(learning_rate=learningRate)
loss = tf.keras.losses.CategoricalCrossentropy()  # Sp
arseCategoricalCrossentropy
metrics = 'categorical_accuracy'  # 's
parse_categorical_accuracy
'
# Configure model
model_1D_CNN.compile(optimizer=optimizer,
                     loss=loss,
                     metrics=[metrics])
# Train
history = model_1D_CNN.fit(X_train, Y_train_OH, epochs=numEpoch, batch_size=
batchSize, validation_split=0.1, verbose=1)