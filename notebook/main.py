from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np

images_dir = "./images/train"
target_size = (64, 64)
target_dims = (64, 64, 3)
n_classes = 2
val_frac = 0.1
batch_size = 32

#, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
data_gen = ImageDataGenerator(validation_split=val_frac)

train_generator = data_gen.flow_from_directory(images_dir, target_size=target_size, batch_size=batch_size, class_mode='binary', subset='training')
val_generator = data_gen.flow_from_directory(images_dir, target_size=target_size, batch_size=batch_size, class_mode='binary', subset='validation')

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=target_dims))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=target_dims))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(activation='relu', units=128))

model.add(Dense(activation='sigmoid', units=1))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, validation_data=val_generator, epochs=20, steps_per_epoch=100)


""" plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'validation'], loc='lower right') """

test_image = load_img('./images/test/15.jpg', target_size=(64,64))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
train_generator.class_indices

test_image = load_img('./images/test/16.jpg', target_size=(64,64))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
train_generator.class_indices

print(result)