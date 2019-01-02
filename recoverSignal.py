import numpy as np
from tools import scaleUp, scaleDown

def recoverSignal(origX,mask,windowLength,mixedPhase,N):
    """ Recover the signal after applying the ideal ratio mask


    # Arguments
        origX: preprocessed x
        mask: ideal ratio mask
        windowLength: number of samples per window
        mixedPhase: mixed phase saved for reconstruction
        N: audio file has values of type intN
        
    # Returns
        recovered: the recovered audio
        
    """


    # Apply the mask estimated by the DNN
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

    # Because mixed = clean + SNR_factor*noise, rescale to range(-1,1)
    if (np.max(abs(overlapped))>1):
        overlapped = np.divide(overlapped,np.max(abs(overlapped)))
    
    # Scale up
    recovered = scaleUp(overlapped,N)

    return recovered


def recoverSignalStandard(origX,windowLength,mixedPhase,N):
    """ Recover the signal without applying any mask


    # Arguments
        origX: preprocessed x
        windowLength: number of samples per window
        mixedPhase: mixed phase saved for reconstruction
        N: audio file has values of type intN
        
    # Returns
        recovered: the recovered audio        
    """


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


def overlapAdd(windowArray):
    """ Add the windowed signal to reconstruct the original signal sequence.
    Help function used in recoverSignal and recoverSignalStandard.


    # Arguments
        windowArray: Nwindows x windowLength matrix, where the windows are overlapping with 50 %
        
    # Returns
        overlapped: vector with overlap-added signal
    """


    [Nwindows,windowLength]= windowArray.shape 
    Ntot = int((Nwindows + 1)*(windowLength/2))
    overlapped = np.zeros(Ntot)
    j = int(windowLength/2)
    halfLength = int(windowLength/2)
    # Add the first half window manually
    overlapped[0:halfLength] = windowArray[0][0:halfLength]
    for i in range(0,(Nwindows-1)):
        # Add the elements corresponding to the current half window
        overlapped[j:j+halfLength] = np.add(windowArray[i][halfLength:],windowArray[i+1][0:halfLength])
        j+=halfLength

    # Add the last half window manually
    overlapped[j:] = windowArray[Nwindows-1][halfLength:]

    return overlapped   