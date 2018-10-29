import numpy
from tools import scaleUp

def recoverSignal(origX,mask,windowLength,mixedPhase,N):
    # origX is the input to the dnn before stacking
    # dnnOutput is the dnn-estimated ideal ratio mask

    # Apply the mask calculated by the neural network
    masked = origX*mask 

    # Take 10^masked to invert the log compression
    invlog = numpy.power(10,masked)

    # Reconstruct the second half of the fft transformed amplitude sequence
    reversed = invlog[:,::-1]
    reconstructedAmplitude = numpy.concatenate((invlog,reversed[:,1:int(windowLength/2)]),axis=1)
    # Include the mixed phase
    fftArray = numpy.multiply(reconstructedAmplitude,numpy.exp(mixedPhase*1j))
    # Perform ifft
    ifftArray = numpy.apply_along_axis(numpy.fft.ifft,axis=1,arr=fftArray)

    # Keep the real output
    realIfftArray = ifftArray.real    
    
    # Do overlap-add
    overlapped = overlapAdd(realIfftArray)

    # Scale up
    recovered = scaleUp(overlapped,N)


    return recovered

def recoverSignalStandard(origX,windowLength,mixedPhase,N):
    # origX is preprocessed, not stacked
    # no masking here.

    # Take 10^masked to invert the log compression
    invlog = numpy.power(10,origX)

    # Reconstruct the second half of the fft transformed amplitude sequence
    reversed = invlog[:,::-1]
    reconstructedAmplitude = numpy.concatenate((invlog,reversed[:,1:int(windowLength/2)]),axis=1)
    # Include the mixed phase
    fftArray = numpy.multiply(reconstructedAmplitude,numpy.exp(mixedPhase*1j))
    # Perform ifft
    ifftArray = numpy.apply_along_axis(numpy.fft.ifft,axis=1,arr=fftArray)

    # Keep the real output
    realIfftArray = ifftArray.real    
    
    # Do overlap-add
    overlapped = overlapAdd(realIfftArray)

    # Scale up
    recovered = scaleUp(overlapped,N)


    return recovered

def inverseFourier(fftArray):
    #ifftArray = numpy.apply_along_axis(numpy.fft.ifft,axis=1,arr=fftArray)
    ifftArray = numpy.fft.ifft(fftArray,axis=1)
    return ifftArray


def overlapAdd(windowArray):
    # windowArray = NWindows x windowLength matrix, overlapped with 50 %
    [Nwindows,windowLength]= windowArray.shape #will work if it is a numpy array
    Ntot = int(Nwindows*windowLength/2 + windowLength/2)
    overlapped = numpy.zeros(Ntot)
    j = int(windowLength/2)
    halfLength = int(windowLength/2)
    overlapped[0:halfLength] = windowArray[0][0:halfLength]
    for i in range(0,(Nwindows-1)):
        overlapped[j:j+halfLength] = numpy.add(windowArray[i][halfLength:],windowArray[i+1][0:halfLength])
        j+=halfLength

    # Need to include the last row manually
    overlapped[j:] = windowArray[Nwindows-1][halfLength:]

    return overlapped   