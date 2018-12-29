import numpy as np

def scaleDown(a,N):
    """ Scale down from intN to float in [-1,1]
    # N = 16 in all file types used

    # Arguments
        N: int type.  
        a: vector

    # Returns
        Downscaled vector    
    """


    c = np.divide(a, 2.0**(N-1) -1)    
    # Prevent values < -1
    c = list(map(lambda x: max(x,-1.0), c))
    return c


def scaleUp(a,N):
    """ Scale up from  [-1,1] to intN,
    only int16 is used.

    # Arguments
        N: int type.  
        a: vector

    # Returns
        Upscaled vector    
    """


    b = list(map(lambda x : x*(2**(N-1)-1),a))
    if N == 32:
        return np.array(b,dtype= np.int32)
    if N==16: 
        return np.array(b,dtype = np.int16)
    return 0



def stackMatrix(matrix):
    """ Stack the matrix such that five successive observations are stored in each row.
    Use zero padding for the two observations in both ends.

    # Arguments
        matrix: nWindows x nValuesPerWindow

    # Returns
        Stacked matrix with the five and five successive windows in each row.    
    """


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
        stackedMatrix[i]= mapToVector(matrix[i-2:i+3,:])

    # Fill in the two last layers manually
    stackedMatrix[Nrows-2][0:Ncolumns]= matrix[Nrows-4]
    stackedMatrix[Nrows-2][Ncolumns:2*Ncolumns]= matrix[Nrows-3]
    stackedMatrix[Nrows-2][2*Ncolumns:3*Ncolumns]= matrix[Nrows-2]
    stackedMatrix[Nrows-2][3*Ncolumns:4*Ncolumns]= matrix[Nrows-1]

    stackedMatrix[Nrows-1][0:Ncolumns]= matrix[Nrows-3]
    stackedMatrix[Nrows-1][Ncolumns:2*Ncolumns]= matrix[Nrows-2]
    stackedMatrix[Nrows-1][2*Ncolumns:3*Ncolumns]= matrix[Nrows-1]
    return stackedMatrix


def mapToVector(matrix):
    """ Help function used in stackMatrix
    Puts five successive rows in a vector

    # Arguments
        matrix: 5 x nValuesPerWindow
        
    # Returns
        Vector with the 5 rows in the input matrix put successively
    """


    Nrow,Ncol = matrix.shape
    vector = matrix.reshape(1,5*Ncol)
    return vector


def findSNRfactor(cleanAudio,noise,SNRdB):
    """ Find the SNR factor that noise must be multiplied by to obtain the specified
    SNRdB

    # Arguments
        cleanAudio: vector with the speech file
        noise: vector with the noise file
        SNRdB: wanted level of SNR
        
    # Returns
        The calculated factor
    """


    Anoise = findRMS(noise)
    if Anoise == 0:
        print('Dividing by zero!!')

    Aclean = findRMS(cleanAudio)
    ANoise_new = Aclean/(10**(SNRdB/20))
    factor = ANoise_new/Anoise
    return factor




def findRMS(vector):
    """ Fint the RMS of a vector.

    # Arguments
        vector: vector to calculate the RMS of     

    # Returns
        The calculated RMS
    """

    #Cast to a large dtype to prevent negative numbers due to overflow
    return np.sqrt(np.mean(np.power(vector,2,dtype='float64')))

# Calculate the ideal ratio mask
def idealRatioMask(cleanAudioMatrix,noiseMatrix,beta):
    """ Calculate the ideal ratio mask

    # Arguments
        cleanAudioMatrix: matrix with preprocessed speech
        noise: matrix with preprocessed noise
        beta: tuning parameter
        
    # Returns
        The calculated ideal ratio mask
    """


    times, frequencies = noiseMatrix.shape
    IRM = np.zeros(shape = (times,frequencies))
    for t in range(0,times):
        for f in range(0,frequencies):
            #for each time-frequency unit
            speechEnergySquared = np.power(cleanAudioMatrix[t,f],2)
            noiseEnergySquared = np.power(noiseMatrix[t,f],2)
            IRM[t,f]= (speechEnergySquared/(speechEnergySquared + noiseEnergySquared))**beta
    return IRM