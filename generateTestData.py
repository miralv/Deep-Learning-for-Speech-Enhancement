import numpy
import random
from pathlib import Path
import glob
import scipy.io.wavfile

from preprocessing import preprocessing
from tools import stackMatrix, decideSNR, idealRatioMask
from recoverSignal import recoverSignalStandard

# Need to create a generator for use in the DNN
def generateTestData(windowLength,q,N,wantedSize,SNRdB): 
    # Temporary: to save space on computer. In the future, we will do a splitting similar to the noise splitting

    group = 2

    L = int(numpy.floor(windowLength/2))

    print('før clean')
    # Load clean audio
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
        if len(audioFiles)> int(wantedSize*q):
            break
    
    print('før noise')
    # Load noise files
    noiseFile = "C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Audio/Nonspeech_test/n70.wav"
    f_rate,data = scipy.io.wavfile.read(noiseFile)
    # make a noise clip consisting of the same noise
    noiseFiles = numpy.array([],dtype = numpy.int16)
    
    while len(noiseFiles)< int(wantedSize*q):
        noiseFiles = numpy.append(noiseFiles,data)
    N_new = len(noiseFiles) - int(len(noiseFiles)%L)
    noiseFiles = noiseFiles[0:N_new]
    #    data = data[0:N_new]
    #noiseFiles = numpy.array([],dtype = numpy.int16)
    #for file in glob.glob(noiseFolder + "*.wav"):
    #    f_rate,data = scipy.io.wavfile.read(file)
    #    N_new = len(data) - int(len(data)%L)
    #    data = data[0:N_new]
    #    noiseFiles = numpy.append(noiseFiles,data)
    #    if len(noiseFiles)> int(wantedSize/q):
    #        break

    # FOR NOW: let both have the same length
    N_min = min(len(audioFiles),len(noiseFiles))
    print('N_min', N_min)
    audioFiles = audioFiles[0:N_min]
    noiseFiles = noiseFiles[0:N_min]

    # Save before-version:
    filePathSaveClean = Path("C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/clean.wav")
    filePathSaveNoise = Path("C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/noise.wav")
    filePathSaveMixed = Path("C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/mixed.wav")
    scipy.io.wavfile.write(filePathSaveClean,48000,data=audioFiles)
    scipy.io.wavfile.write(filePathSaveNoise,48000,data=noiseFiles)
    #scipy.io.wavfile.write(filePathSaveMixed,48000,data=audioFiles+noiseFiles)

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
    #print('before irm')    
    
    # Calculate IRM
    beta = 1/2
    IRM = idealRatioMask(clean,noise,beta)
    #print('after irm')

    # No need to choose this randomly(?)
    y = IRM
    x = mixed
    mixedPhase = mixedPhase
    # Need to stack x
    xStacked = stackMatrix(x,windowLength)

    # Want to save the mixed audio with wanted snr for comparison
    MixedBefore = recoverSignalStandard(x,windowLength,mixedPhase,N)
    scipy.io.wavfile.write(filePathSaveMixed,16000,data=MixedBefore)

    return [x,xStacked,y,mixedPhase]
    

