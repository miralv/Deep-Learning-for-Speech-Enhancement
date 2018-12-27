import numpy as np
import random
from pathlib import Path
import glob
import scipy.io.wavfile

from preprocessing import preprocessing
from tools import stackMatrix, decideSNR, idealRatioMask, findSNRfactor

#Vil teste om det egentlig blir forskjellig å gange med snr-factor???????

# Need to create a generator for use in the DNN
def generateAudioFromFile(windowLength,q,N,batchSize,SNRdB):
    
    
    L = int(np.floor(windowLength/2))

    ## Load clean audio
    audioFolder = "C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Audio/Speech/Train/"
    audioFiles = np.array([],dtype = np.int16)
    for file in glob.glob(audioFolder + "*.wav"):
        f_audio, data = scipy.io.wavfile.read(file)
        data = data[0:( len(data) - len(data)%windowLength)]
        audioFiles = np.append(audioFiles,data)
        
    
    ## Load noise files
    #noiseFile = "C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Audio/Simplified/Noise/Train/n77.wav"
    #f_noise,data = scipy.io.wavfile.read(noiseFile)
        
    #noiseFiles = np.array([],dtype = np.int16)
    #while len(noiseFiles)< int(f_noise*len(audioFiles)/f_audio):
    #    noiseFiles = np.append(noiseFiles,data)
    
    
    
    noiseFolder = "C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Audio/Noise/Train/"
    noiseFiles = np.array([],dtype = np.int16)
    for file in glob.glob(noiseFolder + "*.wav"):
        f_noise, data = scipy.io.wavfile.read(file)
        data = data[0:( len(data) - len(data)%windowLength)]
        noiseFiles = np.append(noiseFiles,data)
        




    # Perform preprocessing before the while
    # Downsample, scale, Hanning and fft, returned as z = x + iy
    audiofftArray = preprocessing(audioFiles,q,N,windowLength,0)
    amplitudeAudio = np.apply_along_axis(np.absolute,axis=1,arr=audiofftArray)
    noisefftArray = preprocessing(noiseFiles,q,N,windowLength,1)
    amplitudeNoise = np.apply_along_axis(np.absolute,axis=1,arr=noisefftArray)
    

    # Make sure they are of the same size
    Nmin = min(np.shape(audiofftArray)[0],np.shape(noisefftArray)[0])
    audiofftArray = audiofftArray[0:Nmin]
    amplitudeAudio = amplitudeAudio[0:Nmin]
    noisefftArray = noisefftArray[0:Nmin]
    amplitudeNoise = amplitudeNoise[0:Nmin]


    # Obtain desired snr
    SNR_factor = findSNRfactor(audioFiles,noiseFiles,SNRdB)
    mixedfftArray = audiofftArray + SNR_factor*noisefftArray
    mixedPhase = np.apply_along_axis(np.angle,axis=1,arr=mixedfftArray)
    mixed = np.apply_along_axis(np.absolute,axis=1,arr=mixedfftArray)
    amplitudeNoiseAdjusted = SNR_factor*amplitudeNoise

    # Keep only the single sided spectrum
    mixed = mixed[:,0:int(windowLength/2+1)]
    clean = amplitudeAudio[:,0:int(windowLength/2+1)]
    noise = amplitudeNoiseAdjusted[:,0:int(windowLength/2+1)]

    # log10. Add ones to avoid amplification of numbers in (0,1)
    o = np.ones(mixed.shape)
    mixed = np.log10(mixed+o)
    clean = np.log10(clean+o)
    noise = np.log10(noise+o)
    
    


    # Calculate IRM
    beta = 1/2
    IRM = idealRatioMask(clean,noise,beta)

    # Need to select audio clips randomly from audioFiles.
    while True:
        stop = mixed.shape[0]-batchSize
        startIndex = random.randint(0,stop)
        y = IRM[startIndex:startIndex+batchSize,:]
        x = mixed[startIndex:startIndex+batchSize,:]
        xStacked = stackMatrix(x,windowLength)
        
        
        yield xStacked,y
