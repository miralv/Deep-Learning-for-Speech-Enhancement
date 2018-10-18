import numpy
from scipy.signal import decimate
from tools import *

def preprocessing(rawAudio,q,N,windowLength):
    # q: downsampling factor (typically 3)
    # N: audiofile has values in intN
    # windowLength: number of samples in each window
    
    # Downsample
    y_new = decimate(rawAudio,q,ftype="fir")
    # Shift to range [-1,1]
    y = scaleDown(y_new,N)

    # Obtain windows and apply Hanning
    hanningArray = Hanning(y,windowLength)

    # Take the fast fourier transform
    [amplitude,storedPhase] = fourier(hanningArray,windowLength)
    # The second half of the sequence gives no new information
    amplitude = amplitude[:,0:int(windowLength/2+1)]
    return [amplitude,storedPhase]
    #fftArray = fourier(hanningArray)
    #return fftArray

def fourier(hanningArray,windowLength):
    # Take fft of each window, i.e. each row
    fftArray = numpy.fft.fft(hanningArray,axis = 1)#numpy.apply_along_axis(numpy.fft,axis=1,arr=hanningArray)
    # Store the phase
    storedPhase = numpy.apply_along_axis(numpy.angle,axis=1,arr=fftArray)
    amplitude = numpy.apply_along_axis(numpy.absolute,axis=1,arr=fftArray)
    return [amplitude,storedPhase]


def Hanning(y,window_length):
    # Create a Hanning window
    window = numpy.hanning(window_length)

    # Apply it
    # Remove entries s.t. there is an integer number of windows 
    N_use = len(y)-len(y)%window_length
    y_use = y[0:N_use]
    N_windows = int(numpy.floor(2*N_use/window_length)-1)

    hanningArray = numpy.zeros(shape=(N_windows,window_length))
    for i in range(0,N_windows):
        start_ind = int(i * window_length/2) #assuming window length is dividible by two.
        hanningArray[i]= y_use[start_ind:start_ind+window_length]*window

    return hanningArray