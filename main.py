import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import scipy.io.wavfile
import sys
import os
import stat
import datetime
import time
import random
import matplotlib.cm as cm
import pandas as pd 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras import regularizers


path = Path("C:/Users/Mira/source/repos/Prosjektoppgave DNN for Speech Enhancement")
os.chdir(path)

from recoverSignal import inverseFourier, overlapAdd, recoverSignal
from preprocessing import preprocessing
from preprocessingWithGenerator import  generateAudioFromFile
from tools import scaleUp,stackMatrix
from generateTestData import generateTestData


## Define variables
windowLength = 256      # Number of samples in each window
N = 16                  # The audioFiles are of type intN
q = 3                   # Downsampling factor for the clean audio files
SNRdB = 5               # Speech to noise ratio in decibels
batchSize = 300         # Mini batch size used in the DNN
Nepochs = 20            # The number of epochs
stepsPerEpoch = 20      # Number of steps (batches of samples yielded from generator) per epoch
audioFolderTest = "C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Audio/Speech/Test/"
audioFolderValidation = "C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Audio/Speech/Validate/"
noiseFileTest = "C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Audio/Noise/Test/n78.wav"
noiseFileValidation = "C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Audio/Noise/Validate/n77.wav"
    


## DNN
# Specify the dimensions of input layer and output layer
inputDim = int((windowLength/2+1)*5)
outputDim = int(windowLength/2+1)

# Define the DNN
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
              loss='mse')

# Generate validation data and test data
_,xValStacked,yVal,_ =  generateTestData(windowLength,q,N,SNRdB, audioFolderValidation,noiseFileValidation)
xTest,xTestStacked,yTest,mixedPhase = generateTestData(windowLength,q,N,SNRdB, audioFolderTest,noiseFileTest)



batchSizes = [4,16,64,128,256]
colors = cm.rainbow(np.linspace(0, 1, len(batchSizes)))


fig2 = plt.subplots(1,1)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim(1, 20)
plt.ylim(0, 0.25)
plt.xticks(np.arange(5,21,5))
plt.yticks(np.arange(0,0.26,0.05))
#plt.show()

#vil lagre resultater fått ved epoch 1, epoch 10 og epoch 20 
epoch_errors_train = np.zeros((len(batchSizes),3))
epoch_errors_validate = np.zeros((len(batchSizes),3))
test_errors = np.zeros(len(batchSizes))
indexes = [0, 9,19]

x_values = np.arange(1,21,1)
for i,batchSize in enumerate(batchSizes):
# Fit the model
    history = model.fit_generator(generateAudioFromFile(windowLength,q,N,batchSize,SNRdB), 
                        validation_data=(xValStacked,yVal),
                        steps_per_epoch=stepsPerEpoch, 
                        epochs=Nepochs,
                        verbose=0)
    
    label_train = 'Train, Batch size=' + str(batchSize)
    label_validate = 'Validation, Batch size=' + str(batchSize)

    plt.plot(x_values,history.history['loss'], ls='dashed',color=colors[i], label= label_train)
    plt.plot(x_values,history.history['val_loss'],color=colors[i], label = label_validate)

    
    epoch_errors_train[i] = [history.history['loss'][ind] for ind in indexes]
    epoch_errors_validate[i] = [history.history['val_loss'][ind] for ind in indexes]
    test_errors[i] = model.evaluate(xTestStacked,yTest)
    
plt.legend()


#plt.show()

   
ts =  '29.12_'
#print(ts)


# Test the model on test data
filePathSave = Path("C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Mixed/Simplified")

v = "history" + ts +".pdf"
savePath = filePathSave / v
plt.savefig(savePath)




predictedY = model.predict(xTestStacked,batch_size=batchSize,verbose=0)
recovered = recoverSignal(xTest,predictedY,windowLength,mixedPhase,N)
trueIRM = recoverSignal(xTest,yTest,windowLength,mixedPhase,N)



## Save for listening
v = "enhancedMain" + str(SNRdB) + ts + ".wav"
savePath = filePathSave / v
scipy.io.wavfile.write(savePath,16000,data=recovered)


# Want to save the applied true irm also:
v = "appliedTrueIRM" + str(SNRdB) + ts + ".wav"
savePath = filePathSave / v
scipy.io.wavfile.write(savePath,16000,data=trueIRM)


## Plot and save plot
s = "IRM" + str(SNRdB) + ts + ".pdf"
savePathPlot = filePathSave / s
#plt.orientation = 'horizontal'
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,5),tight_layout=True)
im1 = ax1.imshow(predictedY.transpose())
ax1.invert_yaxis()
im2 = ax2.imshow(yTest.transpose())
ax2.invert_yaxis()

ax1.set_xlabel('Frame')
ax2.set_xlabel('Frame')
ax1.set_ylabel('Frequency bin')
ax2.set_ylabel('Frequency bin')
ax1.set_title("Predicted IRM")
ax2.set_title("IRM")
im1.set_clim(0,1)
im2.set_clim(0,1)
plt.savefig(savePathPlot)
plt.show()


