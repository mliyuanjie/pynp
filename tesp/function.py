import numpy as np 
from math import pi 


class Nanopore:
    def __init__(self, p) -> None:
        self.p = p 
        self.m_o = np.linspace(0.999, 0.001, 999) 
        m2 = np.power(self.m_o, 2)
        self.y_o = 1/(self.m_o*np.arccos(self.m_o)/np.power(1-m2,1.5)-m2/(1-m2))
        self.m_p = np.linspace(51.95, 1.05, 999) 
        m2 = np.power(self.m_p, 2)
        #interpolate solve the y and m function
        self.y_p = 1/(m2/(m2-1)-self.m_p*np.arccosh(self.m_p)/np.power(m2-1, 1.5))
        self.efield = p["voltage"]*(p["resistivity"]*p["length"]/(pi*p["radius"]*p["radius"])) \
        / (p["resistivity"]*p["length"]/(pi*p["radius"]*p["radius"])+p["resistivity"] \
        / (2*p["radius"]))/p["length"]
        self.g = 1/(pi*p["radius"]*p["radius"]*(p["length"]+1.6*p["radius"])) 

    def __call__(self, imin, imax, i0):
        if (imin == 0) and (imax ==0):
            return {"shape_o":0.0, "volume_o":0.0, "shape_p":0.0, "volume_p":0.0} 
        F_max_o = imax/imin+0.5
        F_min_p = imin/imax+0.5
        index = np.searchsorted(self.y_o, F_max_o, side= 'right')
        if(index>=999):
            index = 998; 
        shape_o = self.m_o[index]
        index = np.searchsorted(self.y_p, F_min_p, side = 'right')
        if(index>=999):
            index =998
        shape_p = self.m_p[index]
        volume_o = imax / (self.g * F_max_o * 1e-27 * i0)
        volume_p = imin / (self.g * F_min_p * 1e-27 * i0)
        return {"shape_o":shape_o, "volume_o":volume_o, "shape_p":shape_p, "volume_p":volume_p} 

    def setPhysical(self, p):
        self.p = p
        self.efield = p["voltage"]*(p["resistivity"]*p["length"]/(pi*p["radius"]*p["radius"])) \
        / (p["resistivity"]*p["length"]/(pi*p["radius"]*p["radius"])+p["resistivity"] \
        / (2*p["radius"]))/p["length"]
        self.g = 1/(pi*p["radius"]*p["radius"]*(p["length"]+1.6*p["radius"]))

    def _getEfield(self):
        return self.efield
