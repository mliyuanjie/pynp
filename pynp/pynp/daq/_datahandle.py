from typing import NamedTuple
from datetime import datetime
from multiprocessing import Queue
from threading import Thread
import io
import numpy as np 
import math
import os
from collections import deque 
from ..cfunction import FindPeakRealTime

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

class Eventlist(NamedTuple):
    start: int 
    end: int 
    start_time: float 
    end_time: float 
    baseline: float 
    current: float 
    begin: float 
    shape_o_rt: float 
    volume_o_rt: float 
    shape_p_rt: float 
    volume_p_rt: float 


class info(NamedTuple):
    data : list
    event: list
    mean: float 
    stdev: float
    efs: int 
    eventstring: str
    v: float

class DaqIOHandle:
    def __init__(self, filename = None, mode = 0, vname = None):
        self.f = None
        self.csv = None
        self.csvstate = False
        self.append = None
        self.appendevent = None
        self.voltagename = "" if vname is None else f'_{int(vname)}mV'
        self.mode = mode
        self.filename = filename
        if filename is None:
            self.append = self.append0
            self.appendevent = self.append0
            return 
        if mode == 0:
            self.append = self.append1
            self.appendevent = self.appendevent1
            self.suffix = '.dat'
        elif mode == 1:
            self.append = self.append2
            self.appendevent = self.appendevent1
            self.suffix = '.dat2'
        else:
            self.append = self.append3
            self.appendevent = self.appendevent1
            self.suffix = '.dat2'
        currenttime = datetime.now()
        currenttime = currenttime.strftime("_%H_%M_%S")
        fn= f'{filename}{currenttime}{self.voltagename}{self.suffix}'
        self.f = open(fn, 'wb')
        self.csv = open(f'{filename}{currenttime}{self.voltagename}{".csv"}', 'w')
        self.csv.write("start,end,start(ms),end(ms),I0(pA),I1(pA),begin,rms(pA)\n")


    def reset(self):
        if self.f is None:
            return
        self.f.close()
        self.csv.close()
        if not self.csvstate:
            os.remove(self.csv.name)
        self.csvstate = False
        currenttime = datetime.now()
        currenttime = currenttime.strftime("_%H_%M_%S")
        filename = f'{self.filename}{currenttime}{self.voltagename}{self.suffix}'
        self.f = open(filename, 'wb')
        self.csv = open(f'{self.filename}{currenttime}{self.voltagename}{".csv"}', 'w')
        self.csv.write("start,end,start(ms),end(ms),I0(pA),I1(pA),begin,rms(pA)\n")

    def setfile(self, filename):
        if self.f is not None:
            self.f.close()
            self.csv.close()
            if not self.csvstate:
                os.remove(self.csv.name)
            self.csvstate = False
            self.f = None 
            self.csv = None
        self.filename = filename
        if filename is None:
            self.append = self.append0
            self.appendevent = self.append0
            return 
        if self.mode == 0:
            self.append = self.append1
            self.appendevent = self.appendevent1
            self.suffix = '.dat'
        elif self.mode == 1:
            self.append = self.append2
            self.appendevent = self.appendevent1
            self.suffix = '.dat2'
        else:
            self.append = self.append3
            self.appendevent = self.appendevent1
            self.suffix = '.dat2'
        currenttime = datetime.now()
        currenttime = currenttime.strftime("_%H_%M_%S")
        fn= f'{filename}{currenttime}{self.voltagename}{self.suffix}'
        self.f = open(fn, 'wb')
        self.csv = open(f'{filename}{currenttime}{self.voltagename}{".csv"}', 'w')
        self.csv.write("start,end,start(ms),end(ms),I0(pA),I1(pA),begin,rms(pA)\n")

    def close(self):
        if self.f is None:
            return
        self.f.close()
        self.csv.close()
        if not self.csvstate:
            os.remove(self.csv.name)

    def append1(self, datai, datav):
        datai[0].astype('float32').tofile(self.f) 

    def append2(self, datai, datav):
        np.transpose(datai.astype('float32')).tofile(self.f) 

    def append3(self, datai, datav):
        repeatn = int(datai[0].size/datav.size)
        v = np.repeat(datav, repeatn).astype('float32') 
        i = datai[0].astype('float32') 
        np.column_stack((i, v)).tofile(self.f) 
    
    def append0(self, datai = None, datav =None):
        pass

    def setfilenamebyvoltage(self, v):
        self.voltagename = f'_{int(v)}mV'
        if self.f is None:
            return
        self.f.close()
        self.csv.close()
        if not self.csvstate:
            os.remove(self.csv.name)
        self.csvstate = False
        currenttime = datetime.now()
        currenttime = currenttime.strftime("_%H_%M_%S")
        filename = f'{self.filename}{currenttime}{self.voltagename}{self.suffix}'
        self.f = open(filename, 'wb')
        self.csv = open(f'{self.filename}{currenttime}{self.voltagename}{".csv"}', 'w')
        self.csv.write("start,end,start(ms),end(ms),I0(pA),I1(pA),begin,rms(pA)\n")

    def appendevent1(self, event:str):
        if len(event) == 0:
            return
        self.csv.write(event)
        self.csvstate = True



class DaqDataHandle:
    '''
    Process the data fragment for in real-time display during Nanopore data collection.
    Main function: average filter, min-max downsampling filter
    '''
    def __init__(self, fs = 500.0, filter = 50.0, window = 5, sigma = 5, direction = -1, move1 = None, move2 = None) -> None:
        self.fs = fs
        self.l = int(fs / filter) if fs > filter else 1
        self.n = int(fs * 20) - self.l + 1
        self.n2 = int(20/window) if window<20 else 1
        mod = self.n % self.n2
        self.padn = self.n2 - mod if mod > 0 else 0
        self.kernal = np.ones(self.l, dtype='float32') / self.l
        self.sweep = 0
        self.findpeak = None
        self.sigma = sigma
        if move1 is not None and move2 is not None:
            self.findpeak = FindPeakRealTime(float(self.fs), float(filter), sigma, int(move1), int(move2), direction)
        self.offset = 0
        self.interval = 1 / (fs * 1000)
        self.process = self.process1 if self.findpeak is None else self.process2
        self.reset = self.reset1 if self.findpeak is None else self.reset2

    def flush(self, p = None):
        if self.findpeak is None:
            return 
        else:
            self.findpeak.flush()

    def setfindevent(self, isfind):
        if self.findpeak:
            self.findpeak.setState(isfind)

    def process1(self, data:np.ndarray):
        data_t = np.convolve(data[0], self.kernal, 'valid')
        mean = np.mean(data_t) 
        stdev = np.std(data_t)
        condition = data_t > (mean + self.sigma * stdev) 
        efs = np.where(np.any(np.logical_and(~condition[0:-1], condition[1:])))[0].size
        data_t = np.pad(data_t, (0, self.padn), 'edge').reshape(self.n2, -1)
        min = np.min(data_t, axis = 1)
        max = np.max(data_t, axis = 1)
        y = np.empty((2 * self.n2,), dtype=min.dtype)
        y[0::2] = min
        y[1::2] = max
        x = np.linspace(self.offset * self.interval, (self.offset + data[0].size) * self.interval, self.n2)
        x = np.repeat(x, 2)
        v = 0 if data.shape[0]==1 else np.mean(data[1])
        self.offset += data[0].size
        return info([x, y],[[x[0], x[-1]],[mean - self.sigma*stdev, mean - self.sigma*stdev],[x[0], x[-1]],[mean + self.sigma*stdev, mean + self.sigma*stdev]], mean, stdev, efs, "", v)

    def process2(self, data:np.ndarray):
        k = math.floor(len(data[0]) / self.n2)
        t = self.findpeak.append(data[0].astype("float32"), k)
        v = 0 if data.shape[0]==1 else np.mean(data[1])
        return info(*t, v)
        
    def setfilter(self, filter):
        self.l = int(self.fs / filter) if self.fs > filter else 1
        self.n = int(self.fs * 20) - self.l + 1
        mod = self.n % self.n2
        self.padn = self.n2 - mod if mod > 0 else 0
        self.kernal = np.ones(self.l, dtype='float32') / self.l

    def setwindow(self, window):
        self.n2 = int(20/window) if window<20 else 1
        mod = self.n % self.n2
        self.padn = self.n2 - mod if mod > 0 else 0
        self.offset = 0

    def reset1(self):
        self.offset = 0

    def reset2(self):
        self.findpeak.reset()

class DaqFitHandle:
    def __init__(self, dp, lp, resistivity, voltage, que:Queue) -> None:
        self.data = np.empty(0, dtype='float32')
        self.offset = 0

        self.m_o = np.linspace(0.999, 0.001, 999) 
        m2 = np.power(self.m_o, 2)
        self.y_o = 1/(self.m_o*np.arccos(self.m_o)/np.power(1-m2,1.5)-m2/(1-m2))
        self.m_p = np.linspace(51.95, 1.05, 999) 
        m2 = np.power(self.m_p, 2)
        self.y_p = 1/(m2/(m2-1)-self.m_p*np.arccosh(self.m_p)/np.power(m2-1, 1.5))
        self.efield = voltage*(resistivity*lp/(np.pi*dp*dp/4)) \
        / (resistivity*lp/(np.pi*dp*dp/4)+resistivity \
        / dp)/lp
        self.g = 4/(np.pi*dp*dp*(lp+0.8*dp)) 
        self.que = que
        self.events = {"shape_o_rt": [], "volume_o_rt": [], "shape_p_rt": [], "volume_p_rt": [], "dI": [], "dt": []}
        self.parameter_id = "shape_o_rt"
    
    def reset(self):
        self.offset = 0 
        self.data = np.empty(0, dtype='float32')
        self.events = {"shape_o_rt": [], "volume_o_rt": [], "shape_p_rt": [], "volume_p_rt": [], "dI": [], "dt": []}

    def setnanopore(self, para):
        
        voltage = para[3]
        resistivity = para[2]
        lp = para[1]
        dp = para[0]
        self.efield = voltage*(resistivity*lp/(np.pi*dp*dp/4)) \
        / (resistivity*lp/(np.pi*dp*dp/4)+resistivity \
        / dp)/lp
        self.g = 4/(np.pi*dp*dp*(lp+0.8*dp)) 
        self.events = {"shape_o_rt": [], "volume_o_rt": [], "shape_p_rt": [], "volume_p_rt": [], "dI": [], "dt": []}

    def setanalysisid(self, id):
        self.parameter_id = id 
        

    def append(self, data, eventstring: str):
        self.data = np.concatenate((self.data, data[0].astype('float32')), axis = 0)
        if len(eventstring) == 0:
            return    
        events = [x for x in eventstring.split('\n') if len(x)>0]
        for i in events: 
            eventpoint = i.split(',')
            dt = float(eventpoint[3])-float(eventpoint[2])
            I0 = float(eventpoint[4])
            dI = float(eventpoint[5])-I0
            
            start = int(eventpoint[6])
            end = int(eventpoint[1])
            if end-start > 5:
                data = np.copy(self.data[start-self.offset:end-self.offset])
                t = Thread(target=self._proteinfit, args = (data, I0, dt, dI))
                t.start()
            self.data = self.data[end-self.offset:]
            self.offset = end 
            
    
    def _proteinfit(self, data, I0, dt, dI):
        buffer = io.BytesIO()
        self.events["dt"].append(dt)
        self.events["dI"].append(dI)
        res = self.calShapeVolume(data, I0)
        self.events["shape_o_rt"].append(res["shape_o"])
        self.events["volume_o_rt"].append(res["volume_o"])
        self.events["shape_p_rt"].append(res["shape_p"])
        self.events["volume_p_rt"].append(res["volume_p"])
        if len(self.events[self.parameter_id]) % 5 != 0:
            return 
        figure = plt.figure(figsize = (3, 3))
        plt.hist(self.events[self.parameter_id]) 
        figure.savefig(buffer, format = 'png')
        plt.close()
        self.que.put_nowait(("events_plot", [buffer]))
        return 

    def calShapeVolume(self, data, I0):
        Imin = np.percentile((I0 - data) / I0, 30) 
        Imax = np.percentile((I0 - data) / I0, 70) 
        F_max_o = Imax/Imin+0.5
        F_min_p = Imin/Imax+0.5
        index = np.searchsorted(self.y_o, F_max_o, side = 'right') 
        if index>=999:
            index = 998 
        shape_o = self.m_o[index] 
        index = np.searchsorted(self.y_p, F_min_p, side = 'right')
        if index>=999:
            index = 998 
        shape_p = self.m_p[index] 
        volume_o = Imax / (self.g * F_max_o * 1e-27) 
        volume_p = Imin / (self.g * F_min_p * 1e-27) 
        return {"shape_o": shape_o, "volume_o":volume_o, "shape_p":shape_p, "volume_p":volume_p}