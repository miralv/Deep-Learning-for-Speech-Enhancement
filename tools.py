import numpy
#Scale from intN to float in [-1,1]
def scaleDown(a,N):
    c = numpy.divide(a, 2.0**(N-1) -1)
    #because the minimum value was 2**N-1, we might have values < -1.
    c = list(map(lambda x: max(x,-1.0), c))
    return c

#Scale up from [-1,1] to intN
def scaleUp(a,N):
    b = list(map(lambda x : x*(2**(N-1)-1),a))
    if N == 32:
        return numpy.array(b,dtype= numpy.int32)
    if N==16: 
        return numpy.array(b,dtype = numpy.int16)
    return 0


# Stack the matrix s.t. the net sees 5 windows at a time. 

def stackMatrix(matrix,windowLength):
    Nrows,Ncolumns = matrix.shape

    stackedMatrix = numpy.zeros(shape = (Nrows,Ncolumns*5))
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


def mapToVector(matrix):
    Nrow,Ncol = matrix.shape
    vector = matrix.reshape(1,5*Ncol)
    return vector
