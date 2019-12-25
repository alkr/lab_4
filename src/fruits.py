import cv2
import keras
import os
import numpy as np
from time import time
from sys import argv
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, InputLayer, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, MaxPooling2D


def load(path, mode):
	images = []
	labels = []
	classes = os.listdir(path)
	for i in range(120):
		cur_path = path + classes[i] + '/'
		fruits = os.listdir(cur_path)
		for x in fruits:
			image = cv2.imread(cur_path + x)
			image = cv2.resize(image, (50, 50))
			images.append(image)
			labels.append(i)
	tensor = np.array(images)
	if mode == 0:
		tensor = np.transpose(tensor, (0, 3, 1, 2))
	return tensor, labels


def go_fully(train_images, train_labels, test_images, test_labels, epoch, batch, close):

	in_shape = (3, 50, 50)
	size = 50 * 50 * 3
	
	activate_func='sigmoid'

	inp_image = Input(shape = in_shape)
	vec_image = Flatten()(inp_image)
	encoder_res = Dense(close, activation=activate_func)(vec_image)

	inp_decoder = Input(shape=(close,))
	vec_decode = Dense(size, activation=activate_func)(inp_decoder)
	decoder_res = Reshape(in_shape)(vec_decode)
  
	encoder = Model(inp_image, encoder_res, name="encoder")
	decoder = Model(inp_decoder, decoder_res, name="decoder")
	autoencoder = Model(inp_image, decoder(encoder(inp_image)), name="autoencoder")

	autoencoder.compile(optimizer='rmsprop', loss='mean_squared_error')
	start = time()
	autoencoder.fit(train_images, train_images, epochs=epoch, batch_size=batch)
	train_time = time() - start
    
	print('FCNN Autoencoder train_time: ' + str(train_time))

	classes = 120

	model = Sequential()
	for layer in encoder.layers:
		model.add(layer)

	model.add(Dense(classes, activation='softmax'))
    
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
	start = time()
	model.fit(train_images, train_labels, epochs=epoch, batch_size=batch)
	train_time = time() - start
    
	print('FCNN Model train_time ' + str(train_time))

	test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
	print('\nFCNN Test accuracy:', test_acc)


def test_fully(epoch, batch, close):
	train_images, train_labels = load('fruits-360_dataset/fruits-360/Training/', 0)
	test_images, test_labels = load('fruits-360_dataset/fruits-360/Test/', 0)     

	in_shape = train_images.shape
	train_images = train_images.astype('float32') / 255
	train_labels = to_categorical(train_labels)
	
	in_shape = test_images.shape
	test_images = test_images.astype('float32') / 255
	test_labels = to_categorical(test_labels)

	go_fully(train_images, train_labels, test_images, test_labels, epoch, batch, close)	


def go_conv(train_images, train_labels, test_images, test_labels, epoch, batch, close):

	in_shape = (50, 50, 3)
	cards = 64
	activate_func = 'relu'

	inp_image = Input(shape = in_shape)
	conv_encode = Conv2D(cards, (3, 3), activation=activate_func, padding='same')(inp_image)
	encoder_res = MaxPooling2D(2)(conv_encode)
  
	inp_decoder = Input(shape=(25, 25, cards))
	decoder_res = Conv2DTranspose(3, (3, 3), strides=(2,2), padding='same')(inp_decoder)
	encoder = Model(inp_image, encoder_res)
	decoder = Model(inp_decoder, decoder_res)
	autoencoder = Model(inp_image, decoder(encoder(inp_image)))

	autoencoder.compile(optimizer='rmsprop', loss='mean_squared_error')
	start = time()
	autoencoder.fit(train_images, train_images, epochs=epoch, batch_size=batch)
	train_time = time() - start
    
	print('CNN Autoencoder train_time ' + str(train_time))

	classes = 120

	model = Sequential()
	for layer in encoder.layers:
		model.add(layer)
	model.add(Flatten())
	model.add(Dense(close, activation=activate_func))
	model.add(Dense(classes, activation='softmax'))
    
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
	start = time()
	model.fit(train_images, train_labels, epochs=epoch, batch_size=batch)
	train_time = time() - start
    
	print('CNN Model train_time ' + str(train_time))

	test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
	print('\nCNN Test accuracy:', test_acc)


def test_conv(epoch, batch, close):
	train_images, train_labels = load('fruits-360_dataset/fruits-360/Training/', 1)
	test_images, test_labels = load('fruits-360_dataset/fruits-360/Test/', 1)     

	in_shape = train_images.shape
	train_images = train_images.astype('float32') / 255
	train_labels = to_categorical(train_labels)
	
	in_shape = test_images.shape
	test_images = test_images.astype('float32') / 255
	test_labels = to_categorical(test_labels)

	go_conv(train_images, train_labels, test_images, test_labels, epoch, batch, close)	


def main():
	epoch, batch, close = 10, 256, 1700
	test_fully(epoch, batch, close)
	
	epoch, batch, close = 5, 256, 1024
	test_conv(epoch, batch, close)


if __name__ == '__main__':
	main()
