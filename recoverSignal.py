import numpy as np
from tools import scaleUp, scaleDown

def recoverSignal(origX,mask,windowLength,mixedPhase,N):
    # origX is the input to the dnn before stacking
    # dnnOutput is the dnn-estimated ideal ratio mask

    # Apply the mask calculated by the neural network
    masked = origX*mask 

    # Take 10^masked to invert the log compression
    invlog = np.power(10,masked)

    # Must subtract 1 to invert the ones added before taking log
    invlog = invlog - np.ones(invlog.shape)


    # Reconstruct the second half of the fft transformed amplitude sequence
    reversed = invlog[:,::-1]
    reconstructedAmplitude = np.concatenate((invlog,reversed[:,1:int(windowLength/2)]),axis=1)
    # Include the mixed phase
    fftArray = np.multiply(reconstructedAmplitude,np.exp(mixedPhase*1j))
    # Perform ifft
    ifftArray = np.apply_along_axis(np.fft.ifft,axis=1,arr=fftArray)

    # Keep the real output
    realIfftArray = ifftArray.real    
    
    # Do overlap-add
    overlapped = overlapAdd(realIfftArray)

    # Because mixed = clean + SNR_factor*noise, er vi ikke lenger i range(-1,1)
    if (np.max(abs(overlapped))>1):
        overlapped = np.divide(overlapped,np.max(abs(overlapped)))
    


    # Scale up
    recovered = scaleUp(overlapped,N)


    return recovered

def recoverSignalStandard(origX,windowLength,mixedPhase,N):
    # origX is preprocessed, not stacked
    # no masking here.

    # Take 10^masked to invert the log compression
    invlog = np.power(10,origX)

    # Must subtract 1 to invert the ones added before taking log
    invlog = invlog - np.ones(invlog.shape)

    # Reconstruct the second half of the fft transformed amplitude sequence
    reversed = invlog[:,::-1]
    reconstructedAmplitude = np.concatenate((invlog,reversed[:,1:int(windowLength/2)]),axis=1)
    # Include the mixed phase
    fftArray = np.multiply(reconstructedAmplitude,np.exp(mixedPhase*1j))
    # Perform ifft
    ifftArray = np.apply_along_axis(np.fft.ifft,axis=1,arr=fftArray)

    # Keep the real output
    realIfftArray = ifftArray.real    
    
    # Do overlap-add
    overlapped = overlapAdd(realIfftArray)

    # Because mixed = clean + SNR_factor*noise, er vi ikke lenger i range(-1,1)
    max_value = np.max(abs(overlapped))
    if (max_value>1):
        overlapped = np.divide(overlapped,max_value)
    else:
        max_value = 1
    
    # Scale up
    recovered = scaleUp(overlapped,N)


    return recovered,max_value

def inverseFourier(fftArray):
    #ifftArray = np.apply_along_axis(np.fft.ifft,axis=1,arr=fftArray)
    ifftArray = np.fft.ifft(fftArray,axis=1)
    return ifftArray


def overlapAdd(windowArray):
    # windowArray = NWindows x windowLength matrix, overlapped with 50 %
    [Nwindows,windowLength]= windowArray.shape 
    Ntot = int((Nwindows + 1)*(windowLength/2))
    overlapped = np.zeros(Ntot)
    j = int(windowLength/2)
    halfLength = int(windowLength/2)
    overlapped[0:halfLength] = windowArray[0][0:halfLength]
    for i in range(0,(Nwindows-1)):
        overlapped[j:j+halfLength] = np.add(windowArray[i][halfLength:],windowArray[i+1][0:halfLength])
        j+=halfLength

    # Need to include the last row manually
    overlapped[j:] = windowArray[Nwindows-1][halfLength:]

    return overlapped   