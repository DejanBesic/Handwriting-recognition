from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation, BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint


def resize_image(image):
    import tensorflow as tf
    return tf.image.resize_images(image, [56, 56])
    0   0  0  0  0
    1 , 2, 3, 4  0
    4 , 5, 6, 7
    7 , 8, 9, 0

    
    
def build_model():
    row, col, ch = 113, 113, 1
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(row, col, ch)))

    model.add(Lambda(resize_image))

    model.add(Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same', name='conv1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1'))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv2'))  
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2'))

    model.add(Convolution2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv3'))  
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3'))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(512, name='dense1'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, name='dense2')) 
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(50, name='output'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


def train_cnn(train_generator, validation_generator, nb_epoch=8, samples_per_epoch=3268, nb_val_samples=842):
    filepath = "check-{epoch:02d}-{val_loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=False)
    callbacks_list = [checkpoint]
    history_object = model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch,
                                         validation_data=validation_generator,
                                         nb_val_samples=nb_val_samples, nb_epoch=nb_epoch, verbose=1, callbacks=callbacks_list)
