import numpy as np
from pathlib import Path
import glob
import scipy.io.wavfile


from preprocessing import preprocessing
from tools import stackMatrix, idealRatioMask,findSNRfactor
from recoverSignal import recoverSignalStandard

def generateTestData(windowLength,q,N,SNRdB, audioFolder, noiseFile): 
    """ Generate test data and validation data


    # Arguments
        windowLength: number of samples per window
        q: downsampling factor
        N: audio file has values of type intN
        SNRdB: wanted level of SNR given in dB
        audioFolder: folder where the speech files are located 
        noiseFile: noise file
        
    # Returns
        Test or validation data:
        x: preprocessed mixed sound
        xStacked: x stacked to have 5 windows per row
        y: analytical IRM
        mixedPhase: mixed phase saved for reconstruction
        
    """


    L = int(np.floor(windowLength/2))

    # Load clean audio
    audioFiles = np.array([],dtype = np.int16)
    for file in glob.glob(audioFolder + "*.wav"):
        f_audio, data = scipy.io.wavfile.read(file)
        data = data[0:( len(data) - len(data)%windowLength)]    
        audioFiles = np.append(audioFiles,data)
        
    # Load noise files    
    f_noise,data = scipy.io.wavfile.read(noiseFile)
    data = data[0:( len(data) - len(data)%windowLength)]    
    noiseFiles = np.array([],dtype = np.int16)
    while len(noiseFiles)< int(f_noise*len(audioFiles)/f_audio):
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

    # Store the phase for reconstruction
    SNR_factor = findSNRfactor(audioFiles,noiseFiles,SNRdB)
    mixedfftArray = audiofftArray + SNR_factor*noisefftArray
    mixedPhase = np.apply_along_axis(np.angle,axis=1,arr=mixedfftArray)
    mixed = np.apply_along_axis(np.absolute,axis=1,arr=mixedfftArray)
    amplitudeNoiseAdjusted = SNR_factor*amplitudeNoise
    
    # Keep only the single sided spectrum
    mixed = mixed[:,0:int(windowLength/2+1)]
    clean = amplitudeAudio[:,0:int(windowLength/2+1)]
    noise = amplitudeNoiseAdjusted[:,0:int(windowLength/2+1)]

    # Log10
    # Add ones to avoid amplification of values in (0,1)
    o = np.ones(mixed.shape)
    mixed = np.log10(mixed+o)
    clean = np.log10(clean+o)
    noise = np.log10(noise+o)
    
    # Calculate IRM
    beta = 1/2
    IRM = idealRatioMask(clean,noise,beta)

    # Give the return variables their values
    y = IRM
    x = mixed
    mixedPhase = mixedPhase
    # Need to stack x
    xStacked = stackMatrix(x)

    # Store the audio files
    MixedBefore,scaling_factor = recoverSignalStandard(x,windowLength,mixedPhase,N)
    noiseFilesAdjusted = noiseFiles*SNR_factor
    audioFiles_before = np.divide(audioFiles,scaling_factor)
    noiseFiles_before = np.divide(noiseFilesAdjusted,scaling_factor)

    filePathSaveClean = Path("C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Mixed/Simplified/cleanScaled.wav")
    filePathSaveNoise = Path("C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Mixed/Simplified/noiseScaled.wav")
    filePathSaveMixed = Path("C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Mixed/Simplified")

    scipy.io.wavfile.write(filePathSaveClean,48000,data=audioFiles)
    scipy.io.wavfile.write(filePathSaveNoise,20000,data=noiseFiles)

    v = "original" + str(SNRdB) + ".wav"
    savePath = filePathSaveMixed / v
    scipy.io.wavfile.write(savePath,16000,data=MixedBefore)
        
    return [x,xStacked,y,mixedPhase]