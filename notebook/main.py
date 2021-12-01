from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

images_dir = "./images/256"
target_size = (256, 256)
target_dims = (256, 256, 3)
n_classes = 2
val_frac = 0.1
batch_size = 16

#, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
data_gen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, validation_split=val_frac)

train_generator = data_gen.flow_from_directory(images_dir, target_size=target_size, batch_size=batch_size, shuffle=True, subset='training')
val_generator = data_gen.flow_from_directory(images_dir, target_size=target_size, batch_size=batch_size, subset='validation')

model = Sequential()

model.add(Conv2D(256, kernel_size=4, strides=1,
 activation='relu', input_shape=target_dims))
model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))
model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu'))
model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

model.summary()

history = model.fit(train_generator, validation_data=val_generator, epochs=5)

test_image = load_img(f'./images/256/test/4_0_3.jpg', target_size=(256,256))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
train_generator.class_indices
print(result)