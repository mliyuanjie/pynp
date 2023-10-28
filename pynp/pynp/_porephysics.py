import numpy as np 
from math import pi 
from scipy.optimize import curve_fit, minimize
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.signal import medfilt, find_peaks, argrelextrema
import queue 
import threading

def _twoGaussian_CDF(x, *params):
    model = params[3]*(1 + erf((x-params[0])/(params[2]*np.sqrt(2)))) +\
            (1-params[3])*(1 + erf((x-params[1])/(params[2]*np.sqrt(2))))
    return model

def _ecdf(sample):

    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

def _DI_CDFx(datax, Imin, Imax, rms, dipole):
    """
    from Jared Matalab code
    """
    x = np.arange(-3000, 3000, 1)
    Imin = np.floor(Imin)
    Imax = np.floor(Imax)
    yb=np.zeros(x.shape, dtype = np.float64)
    yg=1/np.sqrt(2*np.pi*rms)*np.exp(-0.5*np.square(x/rms))
    c = (x>Imin) & (x<Imax) 
    yb[c] = np.cosh(dipole * 3.33356e-30 * np.sqrt((x[c] - Imin) / (Imax - Imin)) / 4.114333424e-21) / np.sqrt((x[c] - Imax) * (Imin - x[c])) 
    y = np.convolve(yg, yb,'same')
    y = np.cumsum(y)
    if y[-1]!=0:
        y=y/y[-1]
    f=interp1d(x,y)
    return f(datax)

def _DI_CDFy(datax, Imin, Imax, rms, dipole):
    """
    from Jared Matalab code
    """
    x = np.arange(-3000, 3000, 1)
    Imin = np.floor(Imin)
    Imax = np.floor(Imax)
    yb=np.zeros(x.shape, dtype = np.float64)
    yg=1/np.sqrt(2*np.pi*rms)*np.exp(-0.5*np.square(x/rms))
    c = (x>Imin) & (x<Imax) 
    yb[c] = np.cosh(dipole * 3.33356e-30 * np.sqrt((x[c] - Imax) / (Imin - Imax)) / 4.114333424e-21) / np.sqrt((x[c] - Imax) * (Imin - x[c])) 
    y = np.convolve(yg,yb,'same')
    y = np.cumsum(y)
    if y[-1]!=0:
        y=y/y[-1]
    f=interp1d(x,y)
    return f(datax)

def lossfunction(xdata, ydata):
    def _cdfx(x):
        ynewx = _DI_CDFx(xdata, *x) 
        return np.linalg.norm(ynewx -  ydata)

    def _cdfy(x):
        ynewy = _DI_CDFy(xdata, *x) 
        return np.linalg.norm(ynewy -  ydata)
    return (_cdfx, _cdfy)

class Nanopore:
    def __init__(self, voltage = -0.1, poreradius = 10e-9, resistivity = 0.046, porelength = 30e-9, iohandle = None) -> None:
        self.m_o = np.linspace(0.999, 0.001, 999) 
        m2 = np.power(self.m_o, 2)
        self.y_o = 1/(self.m_o*np.arccos(self.m_o)/np.power(1-m2,1.5)-m2/(1-m2))
        self.m_p = np.linspace(51.95, 1.05, 9999) 
        m2 = np.power(self.m_p, 2)
        #interpolate solve the y and m function
        self.y_p = 1/(m2/(m2-1)-self.m_p*np.arccosh(self.m_p)/np.power(m2-1, 1.5))
        """
        self.efield = p["voltage"]*(p["resistivity"]*p["length"]/(pi*p["radius"]*p["radius"])) \
        / (p["resistivity"]*p["length"]/(pi*p["radius"]*p["radius"])+p["resistivity"] \
        / (2*p["radius"]))/p["length"]
        self.g = 1/(pi*p["radius"]*p["radius"]*(p["length"]+1.6*p["radius"])) 
        """
        self.efield = voltage *(resistivity * porelength /(pi * poreradius * poreradius)) \
        / (resistivity * porelength / (pi * poreradius * poreradius) + resistivity \
        / (2 *poreradius)) / porelength
        self.g = 1/(pi * poreradius * poreradius * (porelength + 1.6 * poreradius)) 
        self.que_read : queue.Queue = None
        self.que_write : queue.Queue = None
        self.func_map = {"twoGaussianfit": self.twoGaussianfit}
        self.iohandle = iohandle


    def __call__(self, imin, imax, i0):
        if (imin == 0) and (imax ==0):
            return {"Imin": imin, "Imax": imax, "shape_o":0.0, "volume_o":0.0, "shape_p":0.0, "volume_p":0.0} 
        F_max_o = imax/imin+0.5
        F_min_p = imin/imax+0.5
        index = np.searchsorted(self.y_o, F_max_o, side= 'right')
        if(index>=999):
            index = 998; 
        shape_o = self.m_o[index]
        index = np.searchsorted(self.y_p, F_min_p, side = 'right')
        if(index>=9999):
            index =9998
        shape_p = self.m_p[index]
        volume_o = imax / (self.g * F_max_o * 1e-27 * i0)
        volume_p = imin / (self.g * F_min_p * 1e-27 * i0)
        return {"Imin": imin, "Imax": imax, "shape_o":shape_o, "volume_o":volume_o, "shape_p":shape_p, "volume_p":volume_p} 

    def setPhysical(self, voltage = -0.1, poreradius = 10e-9, resistivity = 0.046, porelength = 30e-9):
        self.efield = voltage *(resistivity * porelength /(pi * poreradius * poreradius)) \
        / (resistivity * porelength / (pi * poreradius * poreradius) + resistivity \
        / (2 *poreradius)) / porelength
        self.g = 1/(pi * poreradius * poreradius * (porelength + 1.6 * poreradius)) 
        
    def getEfield(self):
        return self.efield

    def twoGaussianfit(self, data: np.ndarray, I0, I0_rms, Imin, Imax, call_back = None, fileorder = None, eventorder = None):
        if (data.size < 50):
            return None
        rms_max = abs(np.std(data) / I0) * 10000 * 2
        np.abs((I0 - data) / I0 * 10000, out = data)
        rms_min = abs(I0_rms / I0) * 10000 / 2
        if(rms_min > rms_max):
            return None
        rms = (rms_min + rms_max) / 2
        imin = np.min(data) 
        imax = np.max(data) 
        cdfx, cdfy = _ecdf(data)
        Imin_init = np.percentile(data, Imin)
        Imax_init = np.percentile(data, Imax)
        popt, _ = curve_fit(_twoGaussian_CDF, cdfx, cdfy, p0 = [Imin_init, Imax_init, rms, 0.5], bounds=[[imin, imin, rms_min, 0.1],[imax, imax, rms_max, 0.9]])
        I0 = abs(I0)
        Imin = popt[0] / 10000 * I0
        Imax = popt[1] / 10000 * I0
        return self(Imin, Imax, I0)

    def convolvefit(self, data: np.ndarray, I0, I0_rms, Imin, Imax, call_back = None, fileorder = None, eventorder = None):
        if (data.size < 50):
            return None
        rms_max = abs(np.std(data) / I0) * 10000 * 2
        np.abs((I0 - data) / I0 * 10000, out = data)
        rms_min = abs(I0_rms / I0) * 10000 / 2
        if(rms_min > rms_max):
            return None
        rms = (rms_min + rms_max) / 2
        imin = np.min(data) 
        imax = np.max(data) 
        cdfx, cdfy = _ecdf(data)
        Imin_init = np.percentile(data, Imin)
        Imax_init = np.percentile(data, Imax)
        
        dipole_init = abs(550 * self.efield)
        dipole_max = abs(3000 * self.efield)
        lossfunc = lossfunction(cdfx, cdfy)
        popt = None
        if np.median(data) < np.mean(data): 
            popt = minimize(lossfunc[0], x0 = [Imin_init, Imax_init, rms, dipole_init], method='Nelder-Mead', bounds=[[imin, imax],[imin, imax],[rms_min, rms_max],[0, dipole_max]])    
        else:
            popt = minimize(lossfunc[1], x0 = [Imin_init, Imax_init, rms, dipole_init], method='Nelder-Mead', bounds=[[imin, imax],[imin, imax],[rms_min, rms_max],[0, dipole_max]])
        I0 = abs(I0)
        Imin = popt.x[0] / 10000 * I0
        Imax = popt.x[1] / 10000 * I0
        return self(Imin, Imax, I0)

    def statisticfit(self, data: np.ndarray, I0, *args):
        np.abs((I0 - data) / I0 * 10000, out = data)
        #if(rms_min + 1> rms_max):
         #   return None

        data = medfilt(data, 5)
        data_diff = np.diff(data)
        stddev = np.std(data_diff)
        peaks,_ = find_peaks(np.abs(data_diff), height = 2 * stddev)
        if peaks.size == 0:
            return None
        I = np.zeros(peaks.size - 1)
        for i in range(peaks.size - 1):
            I[i] = np.mean(data[peaks[i]:peaks[i+1] + 1])
        Imaxindex = argrelextrema(I, np.greater)[0]
        Iminindex = argrelextrema(I, np.less)[0]
        if Iminindex.size == 0 or Imaxindex.size == 0:
            return None
        Imax = np.mean(I[Imaxindex])
        Imin = np.mean(I[Iminindex])
        I0 = abs(I0)
        Imin = Imin / 10000 * I0
        Imax = Imax / 10000 * I0
        return self(Imin, Imax, I0)
        

    def run(self):
        while True:
            message = self.que_read.get()
            if not message:
                return 
            elif not isinstance(message, tuple):
                return 
            else:
                result = self.func_map[message[0]](*message[1:])
                self.que_write(result)





        

