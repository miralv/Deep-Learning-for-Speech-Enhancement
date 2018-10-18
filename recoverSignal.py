import numpy

def inverseFourier(fftArray):
    #ifftArray = numpy.apply_along_axis(numpy.fft.ifft,axis=1,arr=fftArray)
    ifftArray = numpy.fft.ifft(fftArray,axis=1)
    return ifftArray


def overlapAdd(windowArray):
    [Nwindows,windowLength]= windowArray.shape #will work if it is a numpy array
    Ntot = int(Nwindows*windowLength/2 + windowLength/2)
    overlapped = numpy.zeros(Ntot)
    j = int(windowLength/2)
    halfLength = int(windowLength/2)
    overlapped[0:halfLength] = windowArray[0][0:halfLength]
    for i in range(0,(Nwindows-1)):
        overlapped[j:j+halfLength] = numpy.add(windowArray[i][halfLength:],windowArray[i+1][0:halfLength])
        j+=halfLength

    print('hei')
    # Need to include the last row manually
    overlapped[j:] = windowArray[Nwindows-1][halfLength:]

    return overlapped   