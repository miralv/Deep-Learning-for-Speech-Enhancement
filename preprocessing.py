import numpy as np
from scipy.signal import decimate
from tools import *
import librosa
import resampy
# Downsample, scale perform Hanning and apply FFT on the rawAudio
def preprocessing(rawAudio,q,N,windowLength,noise):
    # q: downsampling factor (typically 3)
    # N: audiofile has values in intN
    # windowLength: number of samples in each window

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

def fourier(hanningArray,windowLength):
    # Take fft of each window, i.e. each row
    fftArray = np.fft.fft(hanningArray,axis = 1)#np.apply_along_axis(np.fft,axis=1,arr=hanningArray)
    # Store the phase
    storedPhase = np.apply_along_axis(np.angle,axis=1,arr=fftArray)
    amplitude = np.apply_along_axis(np.absolute,axis=1,arr=fftArray)
    return [amplitude,storedPhase]


def Hanning(y,windowLength):
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
        start_ind = int(i * windowLength/2) #assuming window length is dividible by two.
        hanningArray[i]= y_use[start_ind:start_ind+windowLength]*window

    return hanningArray