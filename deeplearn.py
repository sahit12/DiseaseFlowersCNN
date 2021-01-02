import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

# ImageDataGenerator for augmentation of images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DiseasePlantsCNN(object):

	@staticmethod
	def make_data_directory(base_dir):
		"""
		base_dir --> It takes the base directory for dataset 
							   containing all datasets

		returns --> (train_dir<str>, test_dir<str>, valid_dir<str>)
		"""

		# Check if the base directory exists, if not create
		if not os.path.exists(base_dir):
			print("Dataset Directory does not exist.\
				Creating the dataset directory\n")
			try:
				os.makedirs(base_dir)
			except OSError as e:
				raise("Invalid filepath. Please provide valid filepath\n")

		# Get the train, test, validation directories path
		train_dir = os.path.join(base_dir, "Disease_Flowers\\train")
		test_dir = os.path.join(base_dir, "Disease_Flowers\\test")
		valid_dir = os.path.join(base_dir, "Disease_Flowers\\validation")

		return (train_dir, test_dir, valid_dir)

	@staticmethod
	def get_class(class_dir):
		"""
		Get classes for the model

		returns --> list of classes/labels
		"""
		classes = []
		image_dir = glob.glob(class_dir + "\\*")
		for cl in image_dir:
			classes.append(cl.split("\\")[-1])

		return classes

	@staticmethod
	def data_augmentation(directory, batch_size, img_size,
		class_mode=None, **kwargs):
		"""
		Performs data augmentation to increase the complexity of the model
		to train. It generalizes the model to a better accuracy of identifying
		a label
		"""
		values = ["rescale", "rotation_range", "width_shift_range",
			"height_shift_range","brightness_range", "shear_range",
			"zoom_range", "horizontal_flip", "vertical_flip"]

		for k,v in kwargs.items():
			if k in values:
				continue
			else:
				raise("Invalid paramaters pass.\
					Allowed paramaters: {}".format(values))

		image_gen = ImageDataGenerator(**kwargs)

		data_gen = image_gen.flow_from_directory(
			batch_size=batch_size,
			directory=directory,
			shuffle=True,
			target_size=(img_size,img_size),
			class_mode=class_mode)

		return data_gen

	@staticmethod
	def learn_model(classes, img_shape, epochs, dpcnn_train_aug, dpcnn_test_aug):
		model = tf.keras.models.Sequential([
		    tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu',
		    	input_shape=(img_shape, img_shape, 3)),
		    tf.keras.layers.MaxPooling2D(2, 2),

		    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
		    tf.keras.layers.MaxPooling2D(2, 2),

		    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
		    tf.keras.layers.MaxPooling2D(2, 2),

		    tf.keras.layers.Dropout(0.2),
		    tf.keras.layers.Flatten(),
		    tf.keras.layers.Dense(512, activation='relu'),
		    tf.keras.layers.Dense(len(classes), activation='softmax')
		])

		model.compile(
			optimizer='adam',
			loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			metrics=["accuracy"]
			)

		epochs = epochs

		history = model.fit_generator(
		    dpcnn_train_aug,
		    epochs=epochs,
		    validation_data=dpcnn_test_aug
		)

		return history

BASE_DIR = "C:\\Users\\IAMLEGEND\\Desktop\\ALL_PROJECT\\MLServe\\KaggleAI\\Dataset"
dpcnn_train, dpcnn_test, dpcnn_val = DiseasePlantsCNN.make_data_directory(base_dir=BASE_DIR)
dpcnn_classes = DiseasePlantsCNN.get_class(dpcnn_train)
print(dpcnn_classes)
dpcnn_train_aug = DiseasePlantsCNN.data_augmentation(dpcnn_train, 40, 150,
	'sparse',
	rotation_range=45,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	zoom_range=0.5,
	horizontal_flip=True,
	vertical_flip=True,
	)
dpcnn_val_aug = DiseasePlantsCNN.data_augmentation(dpcnn_val, 40, 150,
	'sparse',
	rescale=1./255)
hist = DiseasePlantsCNN.learn_model(dpcnn_classes, 150, 80, dpcnn_train_aug, dpcnn_val_aug)