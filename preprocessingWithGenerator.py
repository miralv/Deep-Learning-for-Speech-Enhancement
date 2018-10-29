import numpy
import random
from pathlib import Path
import glob
import scipy.io.wavfile

from preprocessing import preprocessing
from tools import stackMatrix, decideSNR, idealRatioMask

# Need to create a generator for use in the DNN
def generateAudioFromFile(windowLength,q,N,batchSize,SNRdB):
    
    # Temporary: to save space on computer. In the future, we will do a splitting similar to the noise splitting
    group = 1
    
    L = int(numpy.floor(windowLength/2))

    ## Load clean audio
    audioFolder = "C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Audio/part_1/group_"
    audioFiles = numpy.array([],dtype = numpy.int16)
    #for group in range(1,13):
    #save memory for now
    
    if group < 10:
        g = "0" + str(group)
    else:
        g = str(group)

    for file in glob.glob(audioFolder + g +"/*.wav"):
        f_rate, data = scipy.io.wavfile.read(file)
        # Make sure that the audio clip contains an integer number of halfWindows
        N_new = len(data) - int(len(data)%L)
        data = data[0:N_new]
        audioFiles = numpy.append(audioFiles,data)
        if len(audioFiles)> 1000000:
            break
    print('før noise')
    
    ## Load noise files
    noiseFolder = "C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Audio/Nonspeech_train/" 
    noiseFiles = numpy.array([],dtype = numpy.int16)
    for file in glob.glob(noiseFolder + "*.wav"):
        f_rate,data = scipy.io.wavfile.read(file)
        N_new = len(data) - int(len(data)%L)
        data = data[0:N_new]
        noiseFiles = numpy.append(noiseFiles,data)
        if len(noiseFiles)> len(audioFiles):
            break

    # FOR NOW: let both have the same length
    N_min = min(len(audioFiles),len(noiseFiles))
    print('N_min', N_min)
    audioFiles = audioFiles[0:N_min]
    noiseFiles = noiseFiles[0:N_min]

    # Perform preprocessing before the while
    # Downsample, scale, Hanning and fft, returned as z = x + iy
    audiofftArray = preprocessing(audioFiles,q,N,windowLength)
    amplitudeAudio = numpy.apply_along_axis(numpy.absolute,axis=1,arr=audiofftArray)
    noisefftArray = preprocessing(noiseFiles,q,N,windowLength)
    amplitudeNoise = numpy.apply_along_axis(numpy.absolute,axis=1,arr=noisefftArray)
    
    # Store the phase for reconstruction
    mixedfftArray = audiofftArray + noisefftArray
    mixedPhase = numpy.apply_along_axis(numpy.angle,axis=1,arr=mixedfftArray)

    # Add amplitudes to obtain desired snr
    amplitudeNoiseAdjusted = decideSNR(amplitudeAudio,amplitudeNoise,SNRdB)
    mixed = amplitudeAudio + amplitudeNoiseAdjusted

    # Keep only the single sided spectrum
    mixed = mixed[:,0:int(windowLength/2+1)]
    clean = amplitudeAudio[:,0:int(windowLength/2+1)]
    noise = amplitudeNoiseAdjusted[:,0:int(windowLength/2+1)]

    # log10
    mixed = numpy.log10(mixed)
    clean = numpy.log10(clean)
    noise = numpy.log10(noise)
    
    # Calculate IRM
    beta = 1/2
    IRM = idealRatioMask(clean,noise,beta)

    # Need to select audio clips randomly from audioFiles. How many are we choosing at a time? Just test something.
    while True:
        stop = mixed.shape[0]-batchSize
        startIndex = random.randint(0,stop)
        y = IRM[startIndex:startIndex+batchSize,:]
        x = mixed[startIndex:startIndex+batchSize,:]
        xStacked = stackMatrix(x,windowLength)
        yield xStacked,y
