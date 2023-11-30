import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adamax
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from Training.architecture.LeNet import LeNet
from Training.architecture.DenseNet121 import DenseNet121
from Training.architecture.ResNet50 import ResNet50
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import backend as K
from keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from osgeo import gdal
from sklearn.model_selection import train_test_split

class training(object):
	def __init__(self, path_to_dataset, save_to_dir_model, lbl, bs, epochs, lr, seed, numclasses, sizeofimage, splitDataset):
		self.path_to_dataset = path_to_dataset
		self.save_to_dir_model = save_to_dir_model
		self.lbl = lbl
		self.bs = bs
		self.epochs = epochs
		self.lr = lr
		self.seed = seed
		self.numclasses = numclasses
		self.sizeofimage = sizeofimage
		self.test_size = splitDataset
		self.model_history = [] 

	def labeled(self, label):
		if label not in self.lbl:
			return 0
		return self.lbl.get(label)



	def train(self):
		path_code = os.getcwd()
		BS = self.bs
		EPOCHS = self.epochs
		INIT_LR = self.lr
		seed = self.seed
		split_test_size = self.test_size
		total_classes = self.numclasses
		data = []
		labels = []
		BandasImagem=None
		#early_stopping = EarlyStopping(monitor='val_accuracy', patience=0)

		imagePaths = sorted(list(paths.list_images(self.path_to_dataset)))
		random.seed(seed)
		random.shuffle(imagePaths)

		for imagePath in imagePaths:
			filename = os.path.join(path_code, imagePath)
			
			raster = gdal.Open(filename)
			if raster.RasterCount == 1:
				image = raster.ReadAsArray()
			else:
				bandasIm = []
				for i in range(raster.RasterCount):
					bandasIm.append(raster.GetRasterBand(i+1).ReadAsArray())
				image = cv2.merge(bandasIm)	

			#image = cv2.imread(filename,-1)
			BandasImagem=raster.RasterCount
			image = img_to_array(image)
			data.append(image)
			label = imagePath.split(os.path.sep)[-2]
			label = self.labeled(label)
			labels.append(label)

		data = np.array(data, dtype="float")
		labels = np.array(labels)
		(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=split_test_size, random_state=seed)
		trainY = to_categorical(trainY, num_classes=total_classes)
		testY = to_categorical(testY, num_classes=total_classes)
	
		aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
		#model = ResNet50.build(width=5, height=5, depth=BandasImagem, classes=total_classes)
		#model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=INIT_LR), metrics=["accuracy"])
		opt = SGD(lr=INIT_LR)
		#model = DenseNet121.build(width=5, height=5, depth=BandasImagem, classes=total_classes)
		model = LeNet.build(width=5, height=5, depth=BandasImagem, classes=total_classes)
		print(total_classes)
		model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
		# Add ModelCheckpoint callback
		modelDir_Tem = r"/home/nico/dev/cicatrizes_certo/treinamento"

		checkpoint = ModelCheckpoint(modelDir_Tem, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
		early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)
		
		H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1, callbacks=[checkpoint, early_stopping])
		self.model_history.append((model, H.history['val_accuracy'][-1]))

		# Keep only the last 5 models in the history
		self.model_history = self.model_history[-10:]

		# Save the best model among the last 5
		best_model, best_acc = max(self.model_history, key=lambda item:item[1])
		best_model.save(self.save_to_dir_model, include_optimizer=False)

		best_epoch = np.argmax(H.history['val_accuracy']) + 1
		print(f'\nBest epoch: {best_epoch} with val_accuracy: {H.history["val_accuracy"][best_epoch-1]}\n')
		
		plt.style.use("ggplot")
		plt.figure()
		N = len(H.history["loss"])
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
		plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
		plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
		plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
		plt.title("Training Loss and Accuracy on dataset")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss/Accuracy")
		plt.legend(loc="lower left")
		plt.savefig("training_results.png")
		
		from sklearn.metrics import ConfusionMatrixDisplay
		#confusion matrix plot per classes
		titles_options = [
			("Confusion matrix, without normalization", None),
			("Normalized confusion matrix", "true"),
		]
		for title, normalize in titles_options:
			disp = ConfusionMatrixDisplay.from_estimator(
				H,
				testX,
				testY,
				display_labels=class_names,
				cmap=plt.cm.Blues,
				normalize=normalize,
			)
			disp.ax_.set_title(title)

			print(title)
			print(disp.confusion_matrix)

		plt.show()

