import numpy as np
# Scale from intN to float in [-1,1]
def scaleDown(a,N):
    c = np.divide(a, 2.0**(N-1) -1)
    #because the minimum value was 2**N-1, we might have values < -1.
    c = list(map(lambda x: max(x,-1.0), c))
    return c

# Scale up from [-1,1] to intN
def scaleUp(a,N):
    b = list(map(lambda x : x*(2**(N-1)-1),a))
    if N == 32:
        return np.array(b,dtype= np.int32)
    if N==16: 
        return np.array(b,dtype = np.int16)
    return 0


# Stack the matrix s.t. the net sees 5 windows at a time. 
def stackMatrix(matrix,windowLength):
    Nrows,Ncolumns = matrix.shape

    stackedMatrix = np.zeros(shape = (Nrows,Ncolumns*5))
    # Fill the first two layers manually

    stackedMatrix[0][2*Ncolumns:3*Ncolumns]= matrix[0]
    stackedMatrix[0][3*Ncolumns:4*Ncolumns]= matrix[1]
    stackedMatrix[0][4*Ncolumns:5*Ncolumns]= matrix[2]

    stackedMatrix[1][1*Ncolumns:2*Ncolumns]= matrix[0]
    stackedMatrix[1][2*Ncolumns:3*Ncolumns]= matrix[1]
    stackedMatrix[1][3*Ncolumns:4*Ncolumns]= matrix[2]
    stackedMatrix[1][4*Ncolumns:5*Ncolumns]= matrix[3]


    for i in range(2,Nrows-2):
        stackedMatrix[i]= mapToVector(matrix[i-2:i+3,:])#må ha +3 da siste element ikke blir med

    # Fill in the two last layers manually
    stackedMatrix[Nrows-2][0:Ncolumns]= matrix[Nrows-4]
    stackedMatrix[Nrows-2][Ncolumns:2*Ncolumns]= matrix[Nrows-3]
    stackedMatrix[Nrows-2][2*Ncolumns:3*Ncolumns]= matrix[Nrows-2] #midten
    stackedMatrix[Nrows-2][3*Ncolumns:4*Ncolumns]= matrix[Nrows-1]

    stackedMatrix[Nrows-1][0:Ncolumns]= matrix[Nrows-3]
    stackedMatrix[Nrows-1][Ncolumns:2*Ncolumns]= matrix[Nrows-2]
    stackedMatrix[Nrows-1][2*Ncolumns:3*Ncolumns]= matrix[Nrows-1]#midten
    return stackedMatrix

# Help function used in stackMatrix
def mapToVector(matrix):
    Nrow,Ncol = matrix.shape
    vector = matrix.reshape(1,5*Ncol)
    return vector

# Calculate current SNR-level
def calculateSNRdB(cleanAudio,noise):
    SNRdB = 10*np.log10((findRMS(cleanAudio)**2)/(findRMS(noise)**2))
    return SNRdB

# Decide wanted SNR-level and change magnitude of the noise vector accordingly
def decideSNR(cleanAudio,noise,SNRdB):
    # returns the noise-vector multiplied by a constant s.t. 
    # the wanted SNR-level is obtained.
    Anoise = findRMS(noise)
    Aclean = findRMS(cleanAudio)
    ANoise_new = Aclean/(10**(SNRdB/20))
    factor = ANoise_new/Anoise
    return factor*noise

#NB!!!!! audio og noise var i motsatt rekkefølge!!!!!
def findSNRfactor(cleanAudio,noise,SNRdB):
    # returns the noise-vector multiplied by a constant s.t. 
    # the wanted SNR-level is obtained.

    Anoise = findRMS(noise)
    if Anoise == 0:
        print('Dividing by zero!!')

    Aclean = findRMS(cleanAudio)
    ANoise_new = Aclean/(10**(SNRdB/20))
    factor = ANoise_new/Anoise
    return factor




# Find RMS of a vector
def findRMS(vector):
    #Cast to a large dtype to prevent negative numbers due to overflow
    return np.sqrt(np.mean(np.power(vector,2,dtype='float64')))

# Calculate the ideal ratio mask
def idealRatioMask(cleanAudioMatrix,noiseMatrix,beta):
    times, frequencies = noiseMatrix.shape
    IRM = np.zeros(shape = (times,frequencies))
    for t in range(0,times):
        for f in range(0,frequencies):
            #for each time-frequency unit
            speechEnergySquared = np.power(cleanAudioMatrix[t,f],2)
            noiseEnergySquared = np.power(noiseMatrix[t,f],2)
            IRM[t,f]= (speechEnergySquared/(speechEnergySquared + noiseEnergySquared))**beta
    return IRM