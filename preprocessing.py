import numpy
from scipy.signal import decimate
from tools import *

# Downsample, scale perform Hanning and apply FFT on the rawAudio
def preprocessing(rawAudio,q,N,windowLength):
    # q: downsampling factor (typically 3)
    # N: audiofile has values in intN
    # windowLength: number of samples in each window
    
    # Downsample
    #print('y before down sampling:', len(rawAudio))
    yd = decimate(rawAudio,q,ftype="fir")
    #print("y after down sampling:",len(y_new))

    # log10
    #clean = numpy.log10(yd)
    
    # Shift to range [-1,1]
    y = scaleDown(yd,N)
    
    # Obtain windows and apply Hanning
    hanningArray = Hanning(y,windowLength)

    # Take the fourier transform, and get it returned in phormat z = x + iy
    fftArray = numpy.fft.fft(hanningArray)
    # Take the fast fourier transform
    #[amplitude,storedPhase] = fourier(hanningArray,windowLength)
    
    # The second half of the sequence gives no new information
    #amplitude = amplitude[:,0:int(windowLength/2+1)]
    
    # Take log10 of the amplitude
    #amplitudeCompressed = numpy.log10(amplitude)

    return fftArray

def fourier(hanningArray,windowLength):
    # Take fft of each window, i.e. each row
    fftArray = numpy.fft.fft(hanningArray,axis = 1)#numpy.apply_along_axis(numpy.fft,axis=1,arr=hanningArray)
    # Store the phase
    storedPhase = numpy.apply_along_axis(numpy.angle,axis=1,arr=fftArray)
    amplitude = numpy.apply_along_axis(numpy.absolute,axis=1,arr=fftArray)
    return [amplitude,storedPhase]


def Hanning(y,windowLength):
    # Create a Hanning window
    window = numpy.hanning(windowLength)

    # Apply it
    # Remove entries s.t. there is an integer number of windows 
    N_use = len(y)-len(y)%windowLength
    y_use = y[0:N_use]
    N_windows = int(numpy.floor(2*N_use/windowLength)-1)
    print(N_windows)
    hanningArray = numpy.zeros(shape=(N_windows,windowLength))
    for i in range(0,N_windows):
        start_ind = int(i * windowLength/2) #assuming window length is dividible by two.
        hanningArray[i]= y_use[start_ind:start_ind+windowLength]*window

    return hanningArray