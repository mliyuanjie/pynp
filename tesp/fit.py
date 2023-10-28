import numpy as np 
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import erf
from statsmodels.distributions.empirical_distribution import ECDF
from lmfit import minimize, Parameters

__all__ = ['NPfit', 'TwoGaussianfit']

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

def binarySearch(data, time):
    s = 0
    e = len(data)
    mid = int((e+s)/2)
    while e -s >= 1:
        if time >= data[mid]["start"] and time <= data[mid]["end"]:
            return data[mid]
        elif time < data[mid]["start"]:
            e = mid
            mid = int((e+s)/2)
        else:
            s = mid+1
            mid = int((e+s)/2)

    return None

def twoGaussian_CDF(x, *params):
    model = params[3]*(1 + erf((x-params[0])/(params[2]*np.sqrt(2)))) +\
            (1-params[3])*(1 + erf((x-params[1])/(params[2]*np.sqrt(2))))
    return model


def NPfit(data, p, E):
    """
    data: 1d np array 
    p: parameter list dict in example 
    E: electric field, shoud be calculated in Nanopore Class
    partly from Jared Matalab code
    data = (I0-data)/I0*10000 for better result, also rms = rms/I0*10000 before running
    please convert Imin and Imax later
    """
    if (data.size < p["dmin"]) or (data.size>p["dmax"]):
        pres = {"Imin":0.0, "Imax":0.0, "rms":0.0, "dipole":0.0}
        return pres
    out = None
    index = 0
    for i in range(len(data)):
        if data[i+1]<=data[i] and data[i+1]<=data[i+2]:
            index = i+1
            break
    data = data[index:]
    imin = np.min(data)
    imax = np.max(data)
    cdf = ECDF(data)
    para = Parameters()
    para.add("Imin", np.percentile(data, p["Imin"]), min = imin, max = imax)
    para.add("Imax", np.percentile(data, p["Imax"]), min = imin, max = imax)
    para.add("rms", p["rms"], min = p["rms_min"], max = p["rms_max"])
    para.add("dipole", p["dipole"]*E, min = p["dipole_min"]*E, max = p["dipole_max"]*E)
    pres = None
    if stats.mode(data)[0][0] < np.mean(data): 
        out = minimize(DI_CDFx, params = para, args = (cdf.x[1:], cdf.y[1:]), method='nelder', nan_policy='omit')
        pres = {"Imin":out.params["Imin"].value, "Imax":out.params["Imax"].value, "rms":out.params["rms"].value, "dipole":out.params["dipole"].value/E}       
    else:
        out = minimize(DI_CDFy, params = para, args = (cdf.x[1:], cdf.y[1:]), method='nelder', nan_policy='omit')
        pres = {"Imin":out.params["Imin"].value, "Imax":out.params["Imax"].value, "rms":out.params["rms"].value, "dipole":out.params["dipole"].value/E*-1}
    return pres

def TwoGaussianfit(data, p):
    """
    gaussian funcion to fit Imin and Imax
    data = (I0-data)/I0*10000 for better result, also rms = rms/I0*10000 before running
    please convert Imin and Imax later
    """
    if (data.size < p["dmin"]) or (data.size>p["dmax"]):
        pres = {"Imin":0.0, "Imax":0.0, "rms":0.0, "Imin_init":0.0, "Imin_init": 0.0, "Imax_init":0}
        return pres
    out = None
    index = 0
    #find the local min 
    for i in range(len(data)):
        if data[i+1]<=data[i] and data[i+1]<=data[i+2]:
            index = i+1
            break

    data = data[index:]
    imin = np.min(data)
    imax = np.max(data)
    cdf = ECDF(data)
    Imin_init = np.percentile(data, p['Imin'])
    Imax_init = np.percentile(data, p['Imax'])
    popt, pcov = curve_fit(twoGaussian_CDF, cdf.x[1:], cdf.y[1:], p0 = [Imin_init, Imax_init, p["rms"], 0.5], bounds=[[imin, imin, p["rms_min"], 0.1],[imax, imax, p["rms_max"], 0.9]])
    pres = {"Imin":popt[0], "Imax":popt[1], "rms":popt[2], "Imin_init":Imin_init, "Imax_init":Imax_init}
    return pres 



def ADEPT(data, p, index, interval, para):
    """
    from Mosaic source code:
    p: peak dict information
    """
    
    
    p["DI/I0"][index] = p["DI"][index] / p["I0"][index]
    if p["end"][index] - p["start"][index] > 5000 or p["end"][index] - p["start"][index] < 10:
        p["dt(ms)"][index] = (p["end"][index]-p["start"][index]) * interval / 1000
        return
    t = np.linspace(0, len(data)*interval, len(data))
    params=Parameters()
    params.add('mu1', value=(p["start"][index] - p["i"][index]) * interval)
    params.add('mu2', value=(p["end"][index] - p["i"][index]) * interval)
    params.add('a', value=(p["DI"][index]))
    params.add('b', value = p["I0"][index])
    params.add('tau1', value = interval, min =0, max = para["rcmax"])
    if para["rcequal"] == 0:
        params.add('tau2', value = interval, expr='tau1', min =0, max = para["rcmax"])
    else:
        params.add('tau2', value = interval, min =0, max = para["rcmax"])
    out = minimize(stepResponseFunc, params = params, args = (t, data))
    resid = 1-(out.chisqr/np.sum(np.square(data-np.mean(data))))
    #if resid <0.6 and len(data)>500:
    #    out = minimize(stepResponseFunc, params = params, args = (t, data), method='nelder')
    #    resid = 1-(out.chisqr/np.sum(np.square(data-np.mean(data))))

    p["start"][index] = int(out.params["mu1"].value / interval + p["i"][index])
    p["end"][index] = int(out.params["mu2"].value / interval + p["i"][index])
    p["dt(ms)"][index] = (out.params["mu2"].value - out.params["mu1"].value)/1000
    p["a"][index] = out.params["a"].value 
    p["b"][index] = out.params["b"].value
    p["RC1"][index] = out.params["tau1"].value
    p["RC2"][index] = out.params["tau2"].value
    p["chisqr"][index] = resid
    p["good"][index] = 1

    return 


def ADEPT2(data, p):
    """
    from Mosaic source code:
    p[0] interval time us
    p[1] start index
    p[2] end index
    p[3] baseline pA
    p[4] RC time
    p[5] const RC
    p[6] max RC
    """
    if len(data)>100000:
        return {"start":p[1],
            "end":p[2],
            "dt(ms)":(p[2] - p[1])/1000*p[0],
            "DI":p[3]-p[7],
            "I0":p[3],
            "DI/I0":(p[3]-p[7])/p[3],
            "RC1":10, 
            "RC2":10, 
            "chisqr":0,
            "good":0}
    Imin = p[7]
    t = np.linspace(0, len(data)*p[0], len(data))
    params=[]
    params.append(p[1] * p[0])
    params.append(p[2] * p[0])
    params.append((p[3]-Imin))
    params.append(p[3])
    params.append(p[4])
    params.append(p[4])
    params_low = [0,0,np.min(data), np.min(data), 0,0]
    params_high = [(len(data)-1)*p[0], (len(data)-1)*p[0], np.max(data), np.max(data), 20,20]
    out = curve_fit(stepfunc,t,data,p0 = params, jac = stepdfunc, factor = 1)
    residuals = stepfunc(t, data, *out[0])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data-np.mean(data))**2)
    resid = 1 - (ss_res / ss_tot)
    res = { 
            "start":int(out[0][0]/p[0]),
            "end":int(out[0][1]/p[0]),
            "dt(ms)":(out[0][1] - out[0][0])/1000,
            "DI":out[0][2],
            "I0":out[0][3],
            "DI/I0":out[0][2]/out[0][3],
            "RC1":out[0][4], 
            "RC2":out[0][5], 
            "chisqr":resid,
            "good":1
    }
    if res["DI"] > 2 * (res["I0"] - Imin):
        res["DI"] = (res["I0"] - Imin)
        res["DI/I0"] = res["DI"]/res["I0"]
        res["good"] = 0
    return res

