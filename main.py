import h5py as hdf
import numpy as np
import scipy.signal as sp
import scipy.io as sio
import matplotlib.pyplot as plt

class SignalProcessor:
    def __init__(self, name):
        self.name = name
        self.start = None
        self.event_duration = None
        self.result = None
        self.W_f = None
        self.w_f = None
        self.R_y_PS = None
        self.r_y_PS = None
        self.R_y = None
        self.Y = None
        self.yValues = None
        self.xValues = None
        self.max_n = None
        self.K = None
        self.path = 'data/' + name + ".hdf5"
        self.file = hdf.File(self.path, "r")
        self.dset = self.file['strain']['Strain']
        self.XStart = self.dset.attrs.get("Xstart")
        self.dX = self.dset.attrs.get("Xspacing")
        self.N = self.dset.attrs.get("Npoints")
        self.f_s = 1 / self.dX
        self.f = ModuleNotFoundError

    def Subdataset(self, startTime = None, endTime = None):
        beginning = int(0)
        if startTime is not None:
            beginning = int((startTime - self.XStart) // self.dX)

        ending = int((endTime - self.XStart) // self.dX)
        self.max_n = ending - beginning
        if self.max_n % 2 == 0: # make sure max_n is uneven
            self.max_n -= 1
        self.f = np.fft.fftshift(np.fft.fftfreq(self.max_n, d=self.dX)) # Hz
        print("Maximum n value: ", self.max_n, " out of ", self.N)
        self.xValues = self.XStart + self.dX * np.arange(self.max_n)
        self.yValues = self.dset[beginning:self.max_n]
        return self.xValues, self.yValues, self.max_n

    def WindowSubdataset(self):
        # idk wht we are plotting here
        window = sp.windows.chebwin(self.max_n, 100)
        self.yValues *= window
        # I dont understand this plot
        # plt.figure(figsize=(7, 5))
        # plt.semilogy(np.fft.fftshift(np.abs(np.fft.fft(self.yValues))))
        # plt.title("Windowed Time Series")
        # plt.grid(True)
        # plt.show()

    def Periodogram(self, xValues = None, yValues = None, max_n = None):
        if xValues is None:
            xValues = self.xValues
        if yValues is None:
            yValues = self.yValues
        if max_n is None:
            max_n = self.max_n
        self.Y = np.fft.fft(yValues)
        self.R_y = self.Y * np.conj(self.Y)
        self.R_y = np.fft.fftshift(np.abs(self.R_y))
        f = np.fft.fftshift(np.fft.fftfreq(max_n, d=self.dX)) # Hz
        return f, self.R_y

    def PeriodogramSmoothing(self, l_factor = 0.01, m = None):
        # TODO: return delta_omega
        M = m
        if m is None:
            M = int(self.max_n * l_factor)

        H = np.zeros(self.max_n)
        start = (self.max_n - M) // 2
        H[start:start + M] = 1 / M
        h = np.fft.fft(H)

        r_y = np.fft.ifft(self.R_y)
        self.r_y_PS = r_y * h
        self.R_y_PS = np.fft.fftshift(np.fft.fft(self.r_y_PS))
        return self.f, self.R_y_PS
        
    
    def PeriodogramAveraging(self, L = 1000, D = 0.5, window = 'Hanning'):
        if window != 'Hanning':
            raise ValueError("Currently only Hanning window is implemented.")
        else:
            window = np.hanning(L) # Window function for each segment
        D = int(L * D) # Segment overlap
        y = self.yValues
        segments = []

        # segmenting the signal and calculating periodograms:
        k = 0
        end = 0
        while end < len(y):
            start = k * (L - D)
            end = start + L
            if end > len(y):
                break
            segment = y[start:end] * window
            Y = np.fft.fft(segment)
            segments.append(Y * np.conj(Y)) 
            k+=1

        # average the periodograms
        segments = np.array(segments)
        R_y_PA = np.mean(segments, axis=0) # includes normalization

        f_PA = np.fft.fftshift(np.fft.fftfreq(L, d=self.dX))            
        R_y_PA = np.fft.fftshift(R_y_PA) 
        self.K = k
        return f_PA, R_y_PA, k
    
    def WhiteningFilter(self):
        freq = np.linspace(0, 1, self.max_n // 2)
        self.w_f = sp.firwin2(self.max_n, freq, 1 / np.sqrt(self.R_y_PS[0:self.max_n // 2]))
        self.W_f = np.fft.fft(self.w_f)
        plt.figure()
        plt.semilogy(self.f, np.fft.fftshift(np.abs(self.W_f)))


    def Whiten(self, start, duration):
        # Get actual values. The event lasts for approximately 200 ms
        self.event_duration = int(duration // self.dX)
        self.start = int((start + self.event_duration / 2) - self.max_n / 2)
        ye = self.dset[start:start + self.max_n]
        Ye = np.fft.fft(ye)
        Result = Ye * self.W_f
        self.result = np.fft.ifft(Result)
        plt.figure()
        plt.semilogy(self.f, np.abs(Ye))
        plt.figure()
        plt.plot(self.result.real)

    def SaveToFile(self, durations, speedfactor = 1):
        half_durations = int(durations//2)
        cropped_result = self.result[int(self.start - half_durations * self.event_duration):int(self.start + half_durations * self.event_duration)].real
        result_max = max(cropped_result)
        cropped_result /= result_max
        plt.figure()
        plt.plot(cropped_result)
        audio_data = (cropped_result * 2 ** 16 - 1).astype(np.int16)
        sio.wavfile.write("Filtered" + self.name + ".wav", rate=int((1 / self.dX) * 1), data=audio_data)

def periodogram_plotting(f, R_y, title = "Periodogram", ylim = (1e-43, 1e-29), labels = None):
    # f and R_y can be lists of arrays
    if not isinstance(f, list):
        f = [f]
    if not isinstance(R_y, list):
        R_y = [R_y]
    plt.figure(figsize=(7, 5))
    for k in range(len(f)):
        plt.semilogy(f[k], R_y[k], label=labels[k] if labels is not None else None)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Strain (strain$^2$/Hz)")
    plt.ylim(ylim)
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(-2048, 2048)
    if labels is not None:
        plt.legend()
    plt.show()

def plotSubdataset(xValues, yValues):
    plt.figure()
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.title("Time Series of the Strain at the Detector")
    plt.grid(True)
    plt.plot(xValues, yValues)
    plt.xlim(xValues[0], xValues[-1])
    plt.show()

def ASD(f, R_y, labels = None):
    if not isinstance(f, list):
        f = [f]
    if not isinstance(R_y, list):
        R_y = [R_y]
    plt.figure(figsize=(7, 5))
    for k in range(len(f)):
        asd = np.sqrt(R_y[k])
        plt.plot(f[k], asd)
    plt.title("Amplitude Spectral Density of the Noise")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Strain (strain/$\\sqrt{Hz}$)")
    plt.xlim(0, 2048)
    if labels is not None:
        plt.legend(labels)
    plt.grid(True)
    plt.tight_layout()
    plt.show()