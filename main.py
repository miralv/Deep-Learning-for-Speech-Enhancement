import numpy
from pathlib import Path
import glob
import scipy.io.wavfile
from loadFiles import collectAudioFiles
import sys
import os

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


from recoverSignal import inverseFourier, overlapAdd
from preprocessing import preprocessing
from tools import scaleUp,stackMatrix


## Define variables
windowLength = 256  # Number of samples in each window
N = 16              # The audioFiles are of type intN
q = 3               # Downsampling factor
batch_size = 128    # How many observations the neural net looks at before updating parameters
epochs = 1#20       # The number of training runs thorugh the data set






path = Path("C:/Users/Mira/source/repos/Prosjektoppgave DNN for Speech Enhancement")
os.chdir(path)

## Load audio files from file, and put them together in one file. 
rawAudio = collectAudioFiles()

#cropper den litt for nå
rawAudio = rawAudio[0:10000]
## Perform preprocessing
# The raw audio file needs to be down sampled, shifted to the range [-1,1] and fourier transformed
[amplitude, phase] = preprocessing(rawAudio,q,N,windowLength)
#preprocessedArray = preprocessing(rawAudio,q,N,windowLength)
# Stack the windows such that one row in stacked contains two windows on both side of window i
stacked = stackMatrix(amplitude,windowLength)

## DNN
# Split the observations in a training set and a test set
XTrain = stacked

# Specify the dimensions of the net
inputDim = int((windowLength/2+1)*5)
outputDim = int(windowLength/2+1)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(inputDim,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(outputDim, activation='sigmoid'))

model.summary()



## Recover signal
#Dette skal nok flyttes inn i recoverSignal 
transformedBack = inverseFourier(preprocessedArray)
overlapped = overlapAdd(transformedBack)
#Need to upscale again
upscaled = scaleUp(overlapped,16)
# For testing: 
# Save for listening
filePathSave = Path("C:/Users/Mira/Documents/NTNU1819/Bokmål/test2.wav")
scipy.io.wavfile.write(filePathSave,16000,data=upscaled)

## Test quality