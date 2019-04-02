import numpy as np
from scipy.signal import decimate
from tools import *
import librosa
import resampy

def preprocessing(rawAudio,q,N,windowLength,noise):
    """ Downsample, scale perform Hanning and apply FFT on the rawAudio

    # Arguments
        rawAudio: audio file
        q: downsampling factor
        N: audio file has values of type intN
        windowLength: number of samples per window
        noise: boolean variable, 1 if noise
        
    # Returns
        Preprocessed audio
    """


    # Downsample
    if noise !=1:
        yd = decimate(rawAudio,q,ftype="fir")
    else:
        orig_sr = 20000
        target_sr = 16000
        yd = resampy.resample(rawAudio, orig_sr, target_sr)
    
    # Shift to range [-1,1]
    y = scaleDown(yd,N)

    # Obtain windows and apply Hanning
    hanningArray = Hanning(y,windowLength)

    # Take the fourier transform, and get it returned in phormat z = x + iy
    fftArray = np.fft.fft(hanningArray)
    
    return fftArray


def Hanning(y,windowLength):
    """ Apply Hanning on input y
    """


    # Create a Hanning window
    window = np.hanning(windowLength)

    # Apply it
    # Remove entries s.t. there is an integer number of windows 
    N_use = len(y)-len(y)%windowLength
    y_use = y[0:N_use]
    N_windows = int(np.floor(2*N_use/windowLength)-1)
    print(N_windows)
    hanningArray = np.zeros(shape=(N_windows,windowLength))
    for i in range(0,N_windows):
        start_ind = int(i * windowLength/2) # Assuming window length is dividible by two.
        hanningArray[i]= y_use[start_ind:start_ind+windowLength]*window

    return hanningArray