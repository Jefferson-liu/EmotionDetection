import cv2
import tensorflow as tf
import scikitplot
import seaborn as sns
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, Nadam, RMSprop,SGD,Adamax
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.utils import image_dataset_from_directory 
from collections import defaultdict
import os
import pandas as pd

def limit_data(data_dir,n=100):
    a=[]
    for i in os.listdir(data_dir):
        for k,j in enumerate(os.listdir(data_dir+'/'+i)):
            if k>n:continue
            a.append((f'{data_dir}/{i}/{j}',i))
    return pd.DataFrame(a,columns=['filename','class'])

train_data_gen = ImageDataGenerator(
    width_shift_range = 0.1, 
    height_shift_range = 0.1, 
    horizontal_flip = True, 
    rescale=1./255, 
    validation_split = 0.2
    )

validation_data_gen = ImageDataGenerator(
    rescale=1./255, 
    validation_split = 0.2
    )

limit_train = limit_data('data/train', 3000)
limit_test = limit_data('data/test', 1000)

limited_train_data = train_data_gen.flow_from_dataframe(
    limit_train,
    target_size = (48,48),
    color_mode = "grayscale",
    class_mode = "categorical",
    batch_size= 64
)

limited_test_data = validation_data_gen.flow_from_dataframe(
    limit_test,
    target_size = (48,48),
    color_mode = "grayscale",
    class_mode = "categorical",
    batch_size= 64
)

model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

early_stopping = EarlyStopping(
    monitor='val_acc',
    min_delta=0.00005,
    patience=20,
    verbose=1,
    restore_best_weights=True,
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_acc',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1,
)

callbacks = [
    early_stopping,
    lr_scheduler,
]


model.compile(loss = 'categorical_crossentropy', optimizer=Nadam(learning_rate = 0.0002, decay = 1e-6), metrics=['acc'])
batch_size = 64
history = model.fit(
    limited_train_data,
    steps_per_epoch=12000 // batch_size,
    epochs = 150,
    callbacks = callbacks,
    validation_data = limited_test_data,
    validation_steps= 3000//batch_size)

model_json = model.to_json()
with open("model.json", 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')

fig , ax = plt.subplots(1,2)
train_acc = history.history['acc']
train_loss = history.history['loss']
fig.set_size_inches(12,4)

ax[0].plot(history.history['acc'])
ax[0].plot(history.history['val_acc'])
ax[0].set_title('Training Accuracy vs Validation Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend(['Train', 'Validation'], loc='upper left')

ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Training Loss vs Validation Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['Train', 'Validation'], loc='upper left')
plt.savefig("trainval.png") 

plt.show()
