from statsmodels.distributions.empirical_distribution import ECDF
from lmfit import minimize, Parameters
import numpy as np 
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import erf
from ._fitfunction import DI_CDFx, DI_CDFy, stepfunc, stepdfunc, stepResponseFunc, twoGaussian_CDF

def twoGaussian_CDF(x, *params):
    model = params[3]*(1 + erf((x-params[0])/(params[2]*np.sqrt(2)))) +\
            (1-params[3])*(1 + erf((x-params[1])/(params[2]*np.sqrt(2))))
    return model


def TwoGaussianfit(data, p):
    """
    gaussian funcion to fit Imin and Imax
    data = (I0-data)/I0*10000 for better result, also rms = rms/I0*10000 before running
    please convert Imin and Imax later
    """
    if (data.size < p["dmin"]) or (data.size>p["dmax"]):
        pres = {"Imin":0.0, "Imax":0.0, "rms":0.0, "Imin_init":0.0, "Imin_init": 0.0, "Imax_init":0}
        return pres
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





