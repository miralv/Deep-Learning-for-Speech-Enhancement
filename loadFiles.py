# Load noise and clean speech files from prespecified folders.
import numpy
from pathlib import Path
import glob
import scipy.io.wavfile

# Get all names of files in the database
# For now, we take all audio clips from person 1. 

def collectAudioFiles(): 
    fileNames = ([])
    for file in glob.glob("C:/Users/Mira/Documents/NTNU1819/Bokmål/Bokmal/NOR_NOR_1/NOR_NOR_1_T*.wav"):
        fileNames = numpy.append(fileNames,file)
    audioFiles = numpy.array([],dtype = numpy.int16)
    for file_name in fileNames:
        f_rate, data = scipy.io.wavfile.read(file_name)
        audioFiles = numpy.append(audioFiles,data)
    #now is all the audio files collected in aufioFiles
    return audioFiles

#FOR TESTING
#want to listen, to hear possible changes
#filePathSave = Path("C:/Users/Mira/Documents/NTNU1819/Bokmål/test_many.wav")
#scipy.io.wavfile.write(filePathSave,48000,audioFiles)