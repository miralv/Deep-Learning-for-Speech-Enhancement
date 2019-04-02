from scipy.signal import decimate
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
#import glob
import scipy.io.wavfile
from scipy import signal
from mpl_toolkits.axes_grid1 import make_axes_locatable
import resampy
import librosa


#example code
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{cmbright}",
         ]
})

plt.rcParams['axes.labelsize'] = 16
#plt.rcParams['axes.labelsize'] = 14 #16 var for de minste figurene
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

#plt.figure(figsize=(4.5, 2.5))
#plt.plot(range(5))
#plt.text(2.5, 2.,size = '12')
#plt.xlabel(r"µ is not $\mu$")
#plt.tight_layout(.5)

##plt.savefig("pgf_texsystem.pdf")

##plt.plot([1, 2, 3, 4])
##plt.ylabel('some numbers')
#plt.show()


#plt.imshow(np.random.random((70,50)));
#plt.rotation = 'horizontal'
#plt.gca().invert_yaxis()
#plt.ylabel('Channel')
#plt.xlabel('Frame')

#plt.savefig('C:/Users/Mira/image.pdf')
#plt.show()
#plt.savefig('nå.jpg')


#plt.imshow(predictedY)
#plt.orientation='horizontal'
#plt.gca().invert_yaxis()
#plt.ylabel('Channel')
#plt.xlabel('Frame')
#plt.show()


#skal plotte litt av en lydfil for å ha i preprocessing flow diagrammet
#audioFile = "C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Audio/part_1/group_01/p1_g01_f1_1_t-a0001.wav"
#f_rate, data = scipy.io.wavfile.read(audioFile)

#plt.plot(data[1000:-1000], color= 'black')
#plt.show()


#tester spektrogram


#yd = decimate(data,3,ftype="fir")

#f,t,Sxx = signal.spectrogram(x=data, fs=48000,window='hanning',nperseg=256,noverlap=128)
#f,t,Sxx = signal.spectrogram(x=data, fs=48000)

#np.min(data)

#plt.pcolormesh(t,f,np.log(Sxx))
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [s]')
#plt.colorbar()
#plt.show()


#plt.pcolormesh(t,f,Sxx)
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [s]')
#plt.colorbar()
#plt.show()


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%",aspect='auto', pad="2%")
    return fig.colorbar(mappable, cax=cax)
#Plo5

#Plot the masks
plotReady = yTest[0:1400].transpose()
fileName = 'IRM2.pdf'
plotEnhanced = predictedY[0:1400].transpose()
fileName2 = 'IRM_enhanced2.pdf'

def plotRatioMask(mat, fileName):
    x0 = 0
    deltax = 200
    x1 = 1410
    y0 = 0
    deltay = 20
    y1 = 132
    fig,ax = plt.subplots()
    im = ax.imshow(mat,aspect='auto')
    ax.invert_yaxis()
    im.set_clim(0,1)
    colorbar(im)
    ax.xaxis.set_ticks(np.arange(x0,x1,deltax))
    ax.yaxis.set_ticks(np.arange(y0,y1,deltay))
    ax.set(xlabel='Frame', ylabel='FFT bin')
    fig.tight_layout()
    plt.savefig(fileName)

plotRatioMask(plotReady,fileName)
plotRatioMask(plotEnhanced,fileName2)
#plt.show()

#Plot spectrograms of clean, noise and mixed
cleanAudioFile ="C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Mixed/Simplified/cleanScaled.wav"
noiseAudioFile ="C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Mixed/Simplified/noiseScaled.wav"
mixedAudioFile ="C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Mixed/Simplified/original5.wav"
enhancedAudioFile ="C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Mixed/Simplified/enhancedMain508.12. try 1.wav"

#need to plot the spectrogram of the reconstructed speech also
#noiseAudioFile ="C:/Users/Mira/Documents/NTNU1819/Prosjektoppgave/Mixed/Simplified/enhancedMain508.12. try 1.wav"

f_clean, clean = scipy.io.wavfile.read(cleanAudioFile)
f_noise, noise = scipy.io.wavfile.read(noiseAudioFile)
f_mixed, mixed = scipy.io.wavfile.read(mixedAudioFile)
f_enhanced, enhanced = scipy.io.wavfile.read(enhancedAudioFile)

#vil downsample til 16000 først

f = 16000
clean = decimate(clean,int(f_clean/f),ftype="fir")
noise = resampy.resample(noise,f_noise,f)


#vil ha samme lengde som mixed.

l = mixed.shape[0]
clean = clean[0:l]
noise = noise[0:l]

f_c,t_c,S_c = signal.spectrogram(x=clean, fs=f,window='hanning',nperseg=256,noverlap=128)
f_n,t_n,S_n = signal.spectrogram(x=noise, fs=f,window='hanning',nperseg=256,noverlap=128)
f_m,t_m,S_m = signal.spectrogram(x=mixed, fs=f,window='hanning',nperseg=256,noverlap=128)
f_e,t_e,S_e = signal.spectrogram(x=enhanced, fs=f,window='hanning',nperseg=256,noverlap=128)


#p0 = 2*10**(-5)
#dette funker hvis jeg tar scaling=spectrum
#S_cInDecibels = librosa.power_to_db(S_c, ref=p0)
#S_nInDecibels = librosa.power_to_db(S_n, ref=p0)

#plt.plot(S_cInDecibels)
#plt.show()

z1 = np.max(S_c)
z1 = np.log10(np.float64(z1))
z2 = np.max(S_n)
z2 = np.log10(np.float64(z2))
z3 = np.max(S_m)
z3 = np.log10(np.float64(z3))
z4 = np.max(S_e)
z4 = np.log10(np.float64(z4))

maxVal = np.max([z1,z2,z3,z4])
minVal = 0

#fig,ax = plt.subplots()
#im = ax.pcolormesh(t_n,f_n,np.log10(S_n))
#ax.set(xlabel='Time [s]', ylabel='Frequency [Hz]')
##cbar = fig.colorbar(im,ax)
##fig.tight_layout()
#im.set_clim(minVal,maxVal)
#cbar = fig.colorbar(im)
#fig.tight_layout()
#plt.savefig('noiseSpectrogram.pdf')

#fig,ax = plt.subplots()
#im = ax.pcolormesh(t_c,f_c,np.log(S_c))
#ax.set(xlabel='Time [s]', ylabel='Frequency [Hz]')
##fig.tight_layout()
#im.set_clim(minVal,maxVal)
#cbar = fig.colorbar(im)
#cbar.set_ticks(np.arange(0,7,1))
#fig.tight_layout()
#plt.savefig('cleanAudioSpectrogram.pdf')

#plt.show()

def plotSpectrogram(minVal,maxVal,time,freq,spec,fileName):
    fig,ax = plt.subplots()
    im = ax.pcolormesh(time,freq,np.log(spec))
    ax.set(xlabel='Time [s]', ylabel='Frequency [Hz]')
    #fig.tight_layout()
    im.set_clim(minVal,maxVal)
    cbar = fig.colorbar(im)
    cbar.set_ticks(np.arange(0,7,1))
    fig.tight_layout()
    plt.savefig(fileName)


plotSpectrogram(minVal,maxVal,t_c,f_c,S_c,'cleanAudioSpectrogram.pdf')
plotSpectrogram(minVal,maxVal,t_n,f_n,S_n,'noiseSpectrogram.pdf')
plotSpectrogram(minVal,maxVal,t_m,f_m,S_m,'mixedSpectrogram.pdf')
plotSpectrogram(minVal,maxVal,t_e,f_e,S_e,'enhancedSpectrogram.pdf')


#cbar.ax.get_yaxis().labelpad = 5
#cbar.ax.set_ylabel('log(Powerdensity))', rotation=270)

#plotter sigmoid 


#plotter relu
def relu(items):
    return list(map(lambda x: max(x,0),items))


def sigmoid(items):
    return list(map(lambda x: np.divide(1,1+np.exp(-x)),items))
 

x0 = -15
x1 = 15
x = np.arange(x0,x1,0.01)
    
y = relu(x)

fig,ax = plt.subplots(figsize = (10,5))
im = ax.plot(x,y)
ax.set(ylabel=r"$g(x)$",xlabel=r"$x$")
ax.xaxis.set_ticks(np.arange(x0,x1+1,5))
ax.yaxis.set_ticks(np.arange(0,x1+1,5))
plt.savefig('relu.pdf')


z = sigmoid(x)
fig,ax = plt.subplots()
im = ax.plot(x,z)
ax.set(ylabel=r"$g(x)$",xlabel=r"$x$")
ax.xaxis.set_ticks(np.arange(x0,x1+1,deltax=5))
#ax.yaxis.set_ticks(np.arange(0,x1+1,deltay=5))
plt.savefig('sigmoid.pdf')
plt.show()



N = 32

def windowHanning(N):
    ns = np.arange(0,N, 1)
    return list(0.5 * (1 - np.cos(2*np.pi*ns/(N-1))))


g = windowHanning(N)
# normalized frequency = cycles per sample
hx = np.arange(0,1,1/N)

plt.plot(hx,g)

plt.show()

fig,ax = plt.subplots()
im = ax.plot(g)
ax.set(ylabel=r"Amplitude",xlabel=r"Time (samples)")
ax.xaxis.set_ticks(np.arange(0,N,5))
ax.yaxis.set_ticks(np.arange(0,1.1,0.5))
plt.savefig('hanning.pdf')
#plt.show()



N =32 # window length
M = 512 #fft length, padding factor 2
hann_window = windowHanning(N)
#hann_window = hann_window.append(np.zeros(M-N))
#fn  = np.arange(0,1, 1/M)
xi = np.fft.rfft(hann_window,n = M)
max_xi = np.max(xi)
spec = 20*np.log10(np.abs(xi)/max_xi)

#clip at -60 db
#indexes = spec> -80
#xs = np.arange(M)/N
freqs = np.fft.rfftfreq(M,d=1/N)
freqs = freqs/np.max(freqs)

fig,ax = plt.subplots()
im = ax.plot(freqs,spec)
ax.set(ylabel=r"Magnitude (dB)",xlabel=r"Normalized frequency ($\times \pi $ rad/sample)")
#ax.xaxis.set_ticks(np.arange(0,1,0.2))
#ax.yaxis.set_ticks(np.arange(0,1.1,0.5))
#ax.set_xlim(0, 8)
#ax.set_ylim(-60, 0)
fig.tight_layout()
plt.savefig('hanning_lobes.pdf')

plt.show()

#plt.plot(freqs,xi)
#plt.xlim = ((0,0.2))
#plt.show()


