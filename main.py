import numpy
from pathlib import Path
import glob
import scipy.io.wavfile
# from loadFiles import collectAudioFiles
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
SNRdB = -40          # Speech to noise ratio in decibels
wantedSize = 100000
## Load audio files from file, and put them together in one file. 
#rawAudio = collectAudioFiles()

##cropper den litt for nå
#rawAudio = rawAudio[0:10000]
### Perform preprocessing
## The raw audio file needs to be down sampled, shifted to the range [-1,1] and fourier transformed
#[amplitudeCompressed, phase] = preprocessing(rawAudio,q,N,windowLength)


## Stack the windows such that one row in stacked contains two windows on both side of window i
#stacked = stackMatrix(amplitudeCompressed,windowLength)

## DNN
# Split the observations in a training set and a test set
#XTrain = stacked

# Specify the dimensions of the net
inputDim = int((windowLength/2+1)*5)
outputDim = int(windowLength/2+1)

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

xVal,xValStacked,yVal,mixedPhase = generateTestData(windowLength,q,N,wantedSize,SNRdB)
#xVal_stacked is ready for dnn


model.fit_generator(generateAudioFromFile(windowLength,q,N,batchSize,SNRdB), 
                    validation_data=(xValStacked,yVal),
                    steps_per_epoch=10, 
                    epochs=5,
                    verbose=1)

## Recover signal
predictedY = model.predict(xValStacked,batch_size=batchSize,verbose=1)


recovered = recoverSignal(xVal,predictedY,windowLength,mixedPhase,N)

## Save for listening
filePathSave = Path("C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/enhancedMain.wav")
scipy.io.wavfile.write(filePathSave,16000,data=recovered)
#filePathSaveOriginal = Path("C:/Users/Mira/Documents/NTNU1819/Bokmål/original.wav")
### Test quality