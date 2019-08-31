from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from tensorflow.python.keras import losses
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras.preprocessing import image


img_width, img_height = 150,150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 10
nb_validation_samples = 5
epochs = 5
batch_size = 2

if K.image_data_format() == 'channels_first':
    input_shape = (3,img_width,img_height)
else:
    input_shape = (img_width,img_height,3)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size = batch_size,
    class_mode='binary')

validation_generator = train_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size = batch_size,
    class_mode='binary')

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.summary()

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(
    loss = 'binary_crossentropy',
    optimizer='resprop',
    metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size )

model.save_weights('first-try.h5')

img_pred = image.load_img('data/validation/dogs/dog4.jpeg',target_size = (150,150))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(imgz-pred,axis=0)

rslt = model.predict(img_pred)
print(rslt)
if rslt[0][0] ==1:
    prediction = "stress"
else:
    prediction = "relieved"

print(prediction)