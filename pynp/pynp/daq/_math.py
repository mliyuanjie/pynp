import numpy as np 



class GenerateWave:
    def __init__(self, period = 1, ofs = 1000, lagtime = 0, increment = 0, amplitude = 100, v0 = 0, maxv = 100, func = "empty"):
        self.period = period
        self.ofs = ofs
        self.n = int(0.02 * self.ofs)
        self.length = int(self.ofs * self.period)
        self.lagn = int(lagtime * self.ofs)
        self.amplitude = amplitude
        if self.lagn > self.length:
            self.lagn = self.length
        self.increment = increment
        self.v0 = v0
        self.maxv = maxv
        self.vadd = self.v0
        self.get = self.__get0
        self.ptr = 0
        self.y = 0
        table = {"sin":self.sin, "triangle":self.triangle, "sawtooth":self.sawtooth}
        self.func = func
        if func != "empty":
            timearray = np.arange(0, self.length, 1, dtype='float64') / self.length
            self.y = table[func](timearray) * amplitude
            self.y = np.concatenate((self.y, self.y[0:self.n]))
            self.get = self.__get1
            if self.lagn > 1:
                self.y[:self.lagn] = 0 
                self.y[self.length:] = 0    

    def __get0(self):
        if self.ptr >= self.length:
            self.ptr = 0
            self.vadd += self.increment
            if self.vadd / self.maxv > 1:
                self.vadd = self.v0 
        s = self.ptr
        e = s + self.n
        self.ptr = e
        y = np.ones(e - s, 'float64')
        i = 0 if self.lagn <= s else self.lagn - s
        y[0:i] = 0
        return y * self.vadd

    def __get1(self):
        if self.ptr >= self.length:
            self.ptr = 0
            self.vadd += self.increment
            if self.vadd / self.maxv > 1:
                self.vadd = self.v0 
        s = self.ptr
        e = s + self.n
        self.ptr = e
        return self.y[s:e] + self.vadd

    def sin(self, data):
        return np.sin(2 * np.pi * data)

    def triangle(self, data):
        y = data * 4
        y = np.absolute(y - 2) - 1
        return y
    
    def sawtooth(self, data):
        y = data * 2
        y = y - 1
        return y
