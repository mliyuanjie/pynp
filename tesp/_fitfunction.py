import numpy as np 
from math import pi 
from scipy.interpolate import interp1d
from scipy.special import erf


def heaviside(x):
    """
    from Mosaic source code
    """
    out=np.array(x)
    out[out==0]=0.5
    out[out<0]=0
    out[out>0]=1
    return out
	

def I_DT(params, t):
    """
    from Mosaic source code
    """
    tau1 = params[4]
    tau2 = params[5]
    mu1 = params[0]
    mu2 = params[1]
    a = params[2]
    b = params[3]
    m1 = (mu1-t)/tau1
    m1[m1>705] = 705
    m2 = (mu2-t)/tau2
    m2[m2>705] = 705
    m1[m1<-705] = -705
    m2[m2<-705] = -705
    t1=(np.exp(m1)-1)*heaviside(t-mu1)
    t2=(1-np.exp(m2))*heaviside(t-mu2)
    return a*( t1+t2 ) + b

def stepResponseFunc(params, t, data):
    """
    from Mosaic source code
    """
    tau1 = params['tau1'].value
    tau2 = params['tau2'].value
    mu1 = params['mu1'].value
    mu2 = params['mu2'].value
    a = params['a'].value
    b = params['b'].value
    m1 = (mu1-t)/tau1
    m1[m1>705] = 705
    m2 = (mu2-t)/tau2
    m2[m2>705] = 705
    m1[m1<-705] = -705
    m2[m2<-705] = -705
    t1=(np.exp(m1)-1)*heaviside(t-mu1)
    t2=(1-np.exp(m2))*heaviside(t-mu2)
    return data - a*( t1+t2 ) - b

def stepfunc(t, *params):
    """
    from Mosaic source code
    """
    mu1 = params[0]
    mu2 = params[1]
    a = params[2]
    b = params[3]
    tau1 = params[4]
    tau2 = params[5]
    m1 = (mu1-t)/tau1
    m2 = (mu2-t)/tau2
    m1[m1>500] = 500
    m2[m2>500] = 500
    m1[m1<-500] = -500
    m2[m2<-500] = -500
    t1=(np.exp(m1)-1)*np.heaviside(t-mu1, 0.5)
    t2=(1-np.exp(m2))*np.heaviside(t-mu2, 0.5)
    return a*( t1+t2 ) + b

def stepdfunc(t, *params):
    """
    from Mosaic source code
    """
    mu1 = params[0]
    mu2 = params[1]
    a = params[2]
    b = params[3]
    tau1 = params[4]
    tau2 = params[5]
    m1 = (mu1-t)/tau1
    m2 = (mu2-t)/tau2
    m1[m1>500] = 500
    m2[m2>500] = 500
    m1[m1<-500] = -500
    m2[m2<-500] = -500
    t1=(np.exp(m1)-1)*np.heaviside(t-mu1, 0.5)
    t2=(1-np.exp(m2))*np.heaviside(t-mu2, 0.5)

    y1 = -1* a * np.exp(m1) * np.heaviside(t - mu1, 0.5) / tau1
    y2 = a * np.exp(m2) * np.heaviside(t - mu2, 0.5) / tau2
    y3 = t1 + t2
    y4 = np.full(t.shape, 1)
    y5 = -1 * a * t1 * m1 / tau1
    y6 = a * t2 * m2 / tau2

    return np.array([y1, y2, y3, y4, y5, y6]).T

def DI_CDFx(parameter, datax, datay):
    """
    from Jared Matalab code
    """
    x = np.arange(-3000, 3000, 1)
    Imin = np.floor(parameter["Imin"].value)
    Imax = np.floor(parameter["Imax"].value)
    rms = parameter["rms"].value
    dipole = parameter["dipole"].value
    yb=np.zeros(x.shape, dtype = np.float64)
    yg=1/np.sqrt(2*np.pi*rms)*np.exp(-0.5*np.square(x/rms))
    c = (x>Imin) & (x<Imax) 
    yb[c] = np.cosh(dipole * 3.33356e-30 * np.sqrt((x[c] - Imin) / (Imax - Imin)) / 4.114333424e-21) / np.sqrt((x[c] - Imax) * (Imin - x[c])) 
    y = np.convolve(yg, yb,'same')
    y = np.cumsum(y)
    if y[-1]!=0:
        y=y/y[-1]
    f=interp1d(x,y)
    return datay - f(datax)

def DI_CDFy(parameter, datax, datay):
    """
    from Jared Matalab code
    """
    x = np.arange(-3000, 3000, 1)
    Imin = np.floor(parameter["Imin"].value)
    Imax = np.floor(parameter["Imax"].value)
    rms = parameter["rms"].value
    dipole = parameter["dipole"].value
    yb=np.zeros(x.shape, dtype = np.float64)
    yg=1/np.sqrt(2*np.pi*rms)*np.exp(-0.5*np.square(x/rms))
    c = (x>Imin) & (x<Imax) 
    yb[c] = np.cosh(dipole * 3.33356e-30 * np.sqrt((x[c] - Imax) / (Imin - Imax)) / 4.114333424e-21) / np.sqrt((x[c] - Imax) * (Imin - x[c])) 
    y = np.convolve(yg,yb,'same')
    y = np.cumsum(y)
    if y[-1]!=0:
        y=y/y[-1]
    f=interp1d(x,y)
    return datay - f(datax)

def DI_PDFx(parameter, datax):
    """
    from Jared Matalab code
    """
    x = np.arange(-3000, 3000, 1)
    Imin = parameter["Imin"]
    Imax = parameter["Imax"]
    rms = parameter["rms"]
    dipole = parameter["dipole"]
    yb=np.zeros(x.shape, dtype = np.float64)
    yg=1/np.sqrt(2*np.pi*rms)*np.exp(-0.5*np.square(x/rms))
    c = (x>Imin) & (x<Imax) 
    yb[c] = np.cosh(dipole * 3.33356e-30 * np.sqrt((x[c] - Imin) / (Imax - Imin)) / 4.114333424e-21) / np.sqrt((x[c] - Imax) * (Imin - x[c])) 
    y = np.convolve(yg, yb,'same')
    Area = np.trapz(y, x)
    if Area!=0:
        y = y / Area
    f=interp1d(x, y)
    return f(datax)

def DI_PDFy(parameter, datax):
    """
    from Jared Matalab code
    """
    x = np.arange(-3000, 3000, 1)
    Imin = parameter["Imin"]
    Imax = parameter["Imax"]
    rms = parameter["rms"]
    dipole = parameter["dipole"]
    yb=np.zeros(x.shape, dtype = np.float64)
    yg=1/np.sqrt(2*np.pi*rms)*np.exp(-0.5*np.square(x/rms))
    c = (x>Imin) & (x<Imax) 
    yb[c] = np.cosh(dipole * 3.33356e-30 * np.sqrt((x[c] - Imax) / (Imin - Imax)) / 4.114333424e-21) / np.sqrt((x[c] - Imax) * (Imin - x[c])) 
    y = np.convolve(yg,yb,'same')
    Area = np.trapz(y, x)
    if Area!=0:
        y = y / Area
    f=interp1d(x, y)
    return f(datax)

def twoGaussian_CDF(x, *params):
    model = params[3]*(1 + erf((x-params[0])/(params[2]*np.sqrt(2)))) +\
            (1-params[3])*(1 + erf((x-params[1])/(params[2]*np.sqrt(2))))
    return model