import numpy
import numpy
from pathlib import Path
import glob
import scipy.io.wavfile
# from loadFiles import collectAudioFiles
import sys
import os

path = Path("C:/Users/Mira/source/repos/Prosjektoppgave DNN for Speech Enhancement")
os.chdir(path)

from preprocessing import preprocessing
from recoverSignal import overlapAdd, scaleUp

# start with sound
wantedSize = 100000
windowLength = 256
q = 3
N = 16

#NÅ FUNGERER DET KUN DERSOM VI BRUKER SCALING OG DROPPER LOG.


# Load clean audio
FilePath = "C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Audio/part_1/group_01/p1_g01_f1_1_t-a0001.wav"
audioFiles = numpy.array([],dtype = numpy.int16)
#for group in range(1,13):
#save memory for now
f_rate, data = scipy.io.wavfile.read(FilePath)
# Make sure that the audio clip contains an integer number of halfWindows

L = int(numpy.floor(windowLength/2))
N_new = len(data) - int(len(data)%L)
data = data[0:N_new]
audioFiles = numpy.append(audioFiles,data)
    

## Perform preprocessing

# Downsample, scale, Hanning and fft, returned as z = x + iy
audiofftArray = preprocessing(audioFiles,q,N,windowLength)
amplitudeAudio = numpy.apply_along_axis(numpy.absolute,axis=1,arr=audiofftArray)
    
# Store the phase for reconstruction
cleanPhase = numpy.apply_along_axis(numpy.angle,axis=1,arr=audiofftArray)


# Keep only the single sided spectrum
clean = amplitudeAudio[:,0:int(windowLength/2+1)]

# Log compress
clean = numpy.log10(clean)
    
# Need to select audio clips randomly from audioFiles. How many are we choosing at a time? Just test something.
startIndex = 0
xPreprocessed = clean[startIndex:startIndex+wantedSize,:]

## Perform postprocessing
#scaled = scaleUp(xPreprocessed,N)

## Take 10^masked to invert the log compression
invlog = numpy.power(10,xPreprocessed)

# Reconstruct the second half of the fft transformed amplitude sequence
reversed = invlog[:,::-1]
#reversed = xPreprocessed[:,::-1]
reconstructedAmplitude = numpy.concatenate((invlog,reversed[:,1:int(windowLength/2)]),axis=1)
# Include the mixed phase
fftArray = numpy.multiply(reconstructedAmplitude,numpy.exp(cleanPhase*1j))#*1j
# Perform ifft
ifftArray = numpy.apply_along_axis(numpy.fft.ifft,axis=1,arr=fftArray)

# Keep the real output
realIfftArray = ifftArray.real    
    
# Do overlap-add
overlapped = overlapAdd(realIfftArray)

# Invert the log compression and scale up
# Take 10^masked to invert the log compression
#invlog = numpy.power(overlapped,10)
numpy.max(overlapped)
numpy.min(overlapped)
recovered = scaleUp(overlapped,N)

#recovered = overlapped.astype('int16')
# listen
## Save for listening
filePathSave = Path("C:/Users/Mira/Documents/NTNU1819/Bokmål/enhanced.wav")
scipy.io.wavfile.write(filePathSave,16000,data=recovered)
