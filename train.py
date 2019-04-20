import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers
from keras.callbacks import ModelCheckpoint
import os
from utils import INPUT_SHAPE, batch_generator

LEARNING_RATE = .0001
BATCH_SIZE = 128
SAMPLES_PER_EPOCH = 20000
NUM_EPOCHS = 10

# set random seed for reproducibility
np.random.seed(0)

def load_data():
	df = pd.read_csv("data/driving_log.csv")

	X = df[['center', 'left', 'right']].values
	y = df['steering'].values

	# split into train/val sets
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=777)

	return (X_train, y_train), (X_val, y_val)

def build_model():
	model = Sequential()
	model.add(layers.Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
	model.add(layers.Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
	model.add(layers.Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
	model.add(layers.Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
	model.add(layers.Conv2D(64, 3, 3, activation='elu'))
	model.add(layers.Conv2D(64, 3, 3, activation='elu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Flatten())
	model.add(layers.Dense(100, activation='elu'))
	model.add(layers.Dense(50, activation='elu'))
	model.add(layers.Dense(10, activation='elu'))
	model.add(layers.Dense(1))
	model.summary()

	return model

def train(model, X_train, y_train, X_val, y_val):
	checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
	model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
	model.fit_generator(batch_generator("data/images/", X_train, y_train, BATCH_SIZE, True), SAMPLES_PER_EPOCH, NUM_EPOCHS, )

if __name__ == "__main__":

	(X_train, y_train), (X_val, y_val) = load_data()
	print("X_train.shape == " + str(X_train.shape))
	print("y_train.shape == " + str(y_train.shape))
	print("X_val.shape == " + str(X_val.shape))
	print("y_val.shape == " + str(y_val.shape))

	model = build_model()
	train(model, X_train, y_train, X_val, y_val)
	