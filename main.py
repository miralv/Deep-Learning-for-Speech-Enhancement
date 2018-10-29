import numpy
from pathlib import Path
import glob
import scipy.io.wavfile
import sys
import os

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


path = Path("C:/Users/Mira/source/repos/Prosjektoppgave DNN for Speech Enhancement")
os.chdir(path)

from recoverSignal import inverseFourier, overlapAdd, recoverSignal
from preprocessing import preprocessing
from preprocessingWithGenerator import  generateAudioFromFile
from tools import scaleUp,stackMatrix
from generateTestData import generateTestData

## Define variables
windowLength = 256  # Number of samples in each window
N = 16              # The audioFiles are of type intN
q = 3               # Downsampling factor
batchSize = 128     # How many observations the neural net looks at before updating parameters
epochs = 1#20       # The number of training runs thorugh the data set
observationsGeneratedPerLoop = 3000
SNRdB = -10          # Speech to noise ratio in decibels
wantedSize = 100000



## DNN
# Specify the dimensions of the net
inputDim = int((windowLength/2+1)*5)
outputDim = int(windowLength/2+1)

# Specify the architecture of the model
model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(inputDim,)))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(outputDim, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', 
              loss='mse',
              metrics=['accuracy'])

# Generate test data
xVal,xValStacked,yVal,mixedPhase = generateTestData(windowLength,q,N,wantedSize,SNRdB)


# Fit the model
model.fit_generator(generateAudioFromFile(windowLength,q,N,batchSize,SNRdB), 
                    validation_data=(xValStacked,yVal),
                    steps_per_epoch=10, 
                    epochs=5,
                    verbose=1)

# Test the model on test data
predictedY = model.predict(xValStacked,batch_size=batchSize,verbose=1)
#predictedY is the mask calculated by the dnn

## Recover signal 
recovered = recoverSignal(xVal,predictedY,windowLength,mixedPhase,N)

## Save for listening
filePathSave = Path("C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/enhancedMain.wav")
scipy.io.wavfile.write(filePathSave,16000,data=recovered)
