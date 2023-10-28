import numpy as np
from math import pi, sqrt
import random
from pynp import randomWalk, randomAngleWalk, randomWalkDt, randomAngleWalkParallel
import copy


def calposition(n, dx, probdirection, xstart, ystart, xmin, xmax, ymax, radius):
    datax = np.zeros(n) + dx
    datax = np.random.binomial(1, 0.5, n)
    datax[datax == 0] = -1
    datax = datax.astype('float')
    datay = np.random.binomial(1, probdirection, n)
    datay[datay == 0] = -1
    datay = datay.astype('float')
    datax[0] = xstart
    datay[0] = ystart
    collision = 0
    finish = None
    for i in range(1, n):
        if datay[i - 1] >= ymax:
            finish = i
            break
        if datax[i - 1] - radius <= xmin:
            datax[i - 1] = xmin + radius
            collision += 1
        elif datax[i - 1] + radius >= xmax:
            datax[i - 1] = xmax - radius
            collision += 1
        datax[i] = datax[i-1] + datax[i] * dx * np.random.normal(0, 1, 1)
        datay[i] = datay[i-1] + datay[i] * dx * np.random.normal(0, 1, 1)
    if finish is None:
        return (datax, datay, collision)
    else:
        if datax[:finish].size == 0:
            return None, collision
        return (datax[:finish], datay[:finish], collision)


def calcumforrdc(n, dangle, angle_start, dipolefield):
    if dipolefield == 0:
        dipolefield = 1e-30
    data = np.zeros(n)
    data[0] = angle_start
    # directions = np.zeros(n)
    for i in range(1, n):
        pdirection = 1 / (1 + np.exp(dipolefield * 3.33356e-30 *
                          (np.cos(data[i-1]-dangle) - np.cos(data[i-1]+dangle))/8.228666848e-21))
        direction = 1
        # print(pdirection)
        if (random.randrange(0, 100) / 100) <= pdirection:
            direction = dipolefield / abs(dipolefield)
        else:
            direction = dipolefield / abs(dipolefield) * -1
        data[i] = data[i-1] + dangle * direction

        # directions[i] = direction
    return data


def Dt_PDF(t, d, l, v, dt):
    y = np.power(l / (4 * pi * d * np.power(t, 3)), 0.5) * \
        np.exp(-1 * np.power(l - v * t, 2) / (4 * d * t))
    return y


def DI_PDFx(Imin, Imax, dipolewithe):
    """
    from Jared Matalab code
    """
    x = np.arange(-3000, 3000, 1)
    y = np.zeros(x.shape, dtype=np.float64)
    c = (x > Imin) & (x < Imax)
    y[c] = np.cosh(dipolewithe * 3.33356e-30 * np.sqrt((x[c] - Imin) /
                   (Imax - Imin)) / 4.114333424e-21) / np.sqrt((x[c] - Imax) * (Imin - x[c]))
    Area = np.trapz(y, x)
    if Area != 0:
        y = y / Area
    return (x[c], y[c])


def DI_PDFy(Imin, Imax, dipolewithe):
    """
    from Jared Matalab code
    """
    x = np.arange(-3000, 3000, 1)
    y = np.zeros(x.shape, dtype=np.float64)
    c = (x > Imin) & (x < Imax)
    y[c] = np.cosh(dipolewithe * 3.33356e-30 * np.sqrt((x[c] - Imax) /
                   (Imin - Imax)) / 4.114333424e-21) / np.sqrt((x[c] - Imax) * (Imin - x[c]))
    Area = np.trapz(y, x)
    if Area != 0:
        y = y / Area
    return (x[c], y[c])


class TraceSimulator:
    def __init__(self, viscosity=9.91e-4):
        self.Imin = None
        self.Imax = None
        self.viscosity = viscosity
        self.shape = None
        self.volume = None
        self.batch = None

    def addProteins(self, m: np.ndarray, volume: np.ndarray, dipole_field: np.ndarray, Dr: np.ndarray | None) -> None:
        """
        volume and diffusion coefficent dived by nanopore volume 
        """
        self.shape = np.array(m, dtype='float32')
        self.volume = np.array(volume, dtype='float32')
        self.dipole_field = np.array(dipole_field, dtype='float32')
        self.batch = self.shape.shape[0]
        self.radius = np.power(3/4/np.pi/self.shape*self.volume, 1/3)
        self.Dr = None
        self.angle0 = np.random.rand(
            2 * self.batch).astype('float32') * 2 * np.pi
        if Dr is not None:
            self.Dr = np.array(Dr, dtype='float32')
        if self.shape < 1:
            g_epi = np.sqrt(np.arctan(1/self.shape**2-1)) / \
                np.sqrt(1/self.shape**2-1)
            self.Dr = 3 * 4.114333424e-21 / (16*np.pi*self.viscosity*(
                self.radius*self.shape)**3)*((2-1/self.shape**2)*g_epi-1)/(1-1/self.shape**4) / 5  # 5 is the empirical coefficient in nanopore enviroment
        elif self.shape > 1:
            g_epi = 1/np.sqrt(1-1/self.shape**2) * \
                np.log(self.shape*(1+np.sqrt(1-1/self.shape**2)))
            self.Dr = 3 * 4.114333424e-21 / (16*np.pi*self.viscosity*(
                self.radius*self.shape)**3)*((2-1/self.shape**2)*g_epi-1)/(1-1/self.shape**4) / 5
        else:
            Dr_alpha = 1 / self.radius**3
            self.Dr = 3 * 4.114333424e-21 / \
                (16*np.pi*self.viscosity)*Dr_alpha / 5

        self.Imin = np.zeros(self.shape.shape, dtype='float32')
        self.Imax = np.zeros(self.shape.shape, dtype='float32')
        yfactor = np.zeros(self.shape.shape, dtype='float32')

        yfactor[self.shape == 1] = 1.5
        shape = self.shape[self.shape < 1]
        m2 = np.power(shape, 2)
        yfactor[self.shape < 1] = 1 / (shape * np.arccos(shape) /
                                       np.power(1-m2, 1.5) - m2 / (1 - m2))
        shape = self.shape[self.shape > 1]
        m2 = np.power(shape, 2)
        yfactor[self.shape > 1] = 1/(m2/(m2-1)-shape *
                                     np.arccosh(shape)/np.power(m2-1, 1.5))
        self.Imax[self.shape <= 1] = self.volume[self.shape <= 1] * \
            (yfactor[self.shape <= 1])
        self.Imin[self.shape <= 1] = self.Imax[self.shape <= 1] / \
            (yfactor[self.shape <= 1] - 0.5)
        self.Imin[self.shape > 1] = self.volume[self.shape > 1] * \
            (yfactor[self.shape > 1])
        self.Imax[self.shape > 1] = self.Imin[self.shape > 1] / \
            (yfactor[self.shape > 1] - 0.5)

    def run3D(self, n=1024, batch=128, dt=2e-6):
        dangle = np.sqrt(self.Dr * 2 * dt / 10)
        data = np.zeros((2 * batch, n), dtype='float32')
        randomAngleWalkParallel(
            data, self.angle0, self.dipole_field, dangle, 10)
        data = np.cos(data)
        data = data[:batch, :] * data[batch:, :]
        oblate = self.shape <= 1
        prolate = self.shape > 1
        data[oblate] = self.Imin[oblate] + \
            (self.Imax[oblate] - self.Imin[oblate]) * np.square(data[oblate])
        data[prolate] = self.Imax[prolate] + \
            (self.Imin[prolate] - self.Imax[prolate]) * \
            np.square(data[prolate])
        return data


class NanoporeSimulator:
    def __init__(self, poreradius=10e-9, porelength=30e-9, resistivity=0.046, voltage=-0.1, viscosity=9.91e-4) -> None:
        # create a nanopore
        self.efield = voltage * (resistivity * porelength / (pi * poreradius * poreradius)) \
            / (resistivity * porelength / (pi * poreradius * poreradius) + resistivity
               / (2 * poreradius)) / porelength
        self.g = 1/(pi * poreradius * poreradius *
                    (porelength + 1.6 * poreradius))
        self.baseline = voltage / (resistivity / (2 * poreradius) +
                                   (resistivity * porelength) / (pi * poreradius * poreradius))
        self.reactionrate = 0
        self.ymax = porelength
        self.xmin = poreradius * -1
        self.xmax = poreradius
        self.viscosity = viscosity
        self.Imin = None
        self.Imax = None
        self.shape = None
        self.volume = None
        self.probon = 0
        self.koff = 0
        self.simulate_t0 = 0
        self.event_list = None
        self.fs_list = None

    def addProtein(self, radiusx=3e-9, radiusy=3e-9, radiusz=2e-9, charge=-5, Dipolement=500, Dr = None):
        """
        standard unit, m, e, m^2/s, rad^2/s, s
        """
        # set eventinterval by concentration
        # set dwell time by charge

        # check oblate or prolate
        sortedshape = [radiusz, radiusx, radiusy]
        aaxis = 0
        sortedshape.sort()
        if sortedshape[1] - sortedshape[0] >= sortedshape[2] - sortedshape[1]:
            self.shape = 2 * sortedshape[0] / (sortedshape[1] + sortedshape[2])
            aaxis = (sortedshape[1] + sortedshape[2]) / 2
        else:
            self.shape = 2 * sortedshape[2] / (sortedshape[0] + sortedshape[1])
            aaxis = (sortedshape[0] + sortedshape[1]) / 2
        self.charge = charge * 1.602176634e-19
        self.dipole = Dipolement
        self.Df = 1
        self.Dr = 1
        if self.shape < 1:
            # epi = np.sqrt(1-self.shape**2)
            # h_epi = 1/epi**3*(np.arcsin(epi)-epi*np.sqrt(1-epi**2))
            # k_epi = (1/(2-epi**2))*(2*np.sqrt(1-epi**2)-(1-2*epi**2)*h_epi)
            # t_epi =  np.sqrt(np.arctan(epi**2/(1-epi**2)))/epi
            # Df_alpha = t_epi / aaxis
            # Dr_alpha = k_epi / aaxis**3
            g_epi = np.sqrt(np.arctan(1/self.shape**2-1)) / \
                np.sqrt(1/self.shape**2-1)
            self.Df = 4.114333424e-21 / \
                (6*np.pi*self.viscosity*aaxis*self.shape) * g_epi
            self.Dr = 3 * 4.114333424e-21 / (16*np.pi*self.viscosity*(
                aaxis*self.shape)**3)*((2-1/self.shape**2)*g_epi-1)/(1-1/self.shape**4)
        elif self.shape > 1:
            # epi = np.sqrt(1-1/self.shape**2)
            # g_epi = (1/(2*epi**3))*(2*epi/(1-epi**2)-np.log((1+epi)/(1-epi)))
            # f_epi = 2/((2-epi**2)*(1-epi**2)) - ((1+epi**2)/(1-epi**2))*g_epi
            # d_epi = np.log((1+epi)/np.sqrt(1-epi**2))/epi
            # Df_alpha = d_epi / (aaxis * self.shape)
            # Dr_alpha = f_epi / (aaxis * self.shape)**3
            g_epi = 1/np.sqrt(1-1/self.shape**2) * \
                np.log(self.shape*(1+np.sqrt(1-1/self.shape**2)))
            self.Df = 4.114333424e-21 / \
                (6*np.pi*self.viscosity*aaxis*self.shape) * g_epi
            self.Dr = 3 * 4.114333424e-21 / (16*np.pi*self.viscosity*(
                aaxis*self.shape)**3)*((2-1/self.shape**2)*g_epi-1)/(1-1/self.shape**4)
        else:
            Df_alpha = 1 / aaxis
            Dr_alpha = 1 / aaxis**3
            self.Df = 4.114333424e-21 / (6*np.pi*self.viscosity) * Df_alpha
            self.Dr = 3 * 4.114333424e-21 / (16*np.pi*self.viscosity)*Dr_alpha

        self.borderwidth = max(sortedshape)
        self.volume = 4/3 * pi * radiusx * radiusy * radiusz
        # calculate Imin and Imax in this nanopore condition
        self.yfactor = None
        self.pdf_event = None
        if self.shape < 1:
            m2 = np.power(self.shape, 2)
            self.yfactor = 1/(self.shape*np.arccos(self.shape) /
                              np.power(1-m2, 1.5)-m2/(1-m2))
            self.Imax = self.volume * (self.g * self.yfactor * self.baseline)
            self.Imin = self.Imax / (self.yfactor - 0.5)
        elif self.shape > 1:
            m2 = np.power(self.shape, 2)
            self.yfactor = 1/(m2/(m2-1)-self.shape *
                              np.arccosh(self.shape)/np.power(m2-1, 1.5))
            self.Imin = self.volume * (self.g * self.yfactor * self.baseline)
            self.Imax = self.Imin / (self.yfactor - 0.5)
        else:
            self.yfactor = 1.5
            self.Imax = self.volume * (self.g * self.yfactor * self.baseline)
            self.Imin = self.Imax / (self.yfactor - 0.5)

        self.dipolefield = self.efield * self.dipole
        self.velocity = self.charge * self.efield * self.Dr / 4.114333424e-21 
        if Dr is not None:
            self.Dr = Dr

    def addReaction(self, bond_length=3e-10, k_off=3e-2, k_on=0.1, reaction_radius=7e-9):
        self.kon = k_on * \
            np.exp(abs(self.efield * self.charge * bond_length) / 4.114333424e-21)
        self.koff = k_off * \
            np.exp(abs(self.efield * self.charge * bond_length) / 4.114333424e-21)
        self.reactradus = reaction_radius

    def simulateDt(self, maxtime=100, dt=1e-9):
        dx = np.sqrt(self.Df * 2 * dt)
        Ppos = np.exp(-1 * self.efield * self.charge * dx / 4.114333424e-21)
        Pneg = np.exp(self.efield * self.charge * dx / 4.114333424e-21)
        probdirection = Pneg / (Ppos + Pneg)
        res = randomWalkDt(maxtime, dt, dx, probdirection, self.xmin,
                           self.xmax, self.ymax, self.reactradus, self.kon, self.koff)
        return res

    def simulateRelativeRMS(self, n=10000, rms=50e-12):
        baseline = np.random.normal(0, rms, n) / self.baseline
        return baseline

    def simulateAngle(self, angle_start_x=0.0, n=10000, dt=2e-6):
        dangle = sqrt(self.Dr * 2 * dt / 5 / 10)
        dipolefield = self.efield * self.dipole
        data_x = randomAngleWalk(n*10, dangle, angle_start_x, dipolefield)
        data_x = np.array(data_x)[::10]
        return data_x

    def simulateIntraCurrent(self, angle_start=0.0, n=10000, dt=2e-6):
        dangle = sqrt(self.Dr * 2 * dt / 5 / 10)
        dipolefield = self.efield * self.dipole
        data = randomAngleWalk(n*10, dangle, angle_start, dipolefield)
        angle = data[-1]
        data = np.array(data)[::10]
        if self.shape < 1:
            data = self.Imin + (self.Imax - self.Imin) * \
                np.square(np.cos(data))
        elif self.shape > 1:
            data = self.Imax + (self.Imin - self.Imax) * \
                np.square(np.cos(data))
        else:
            data = self.Imax + np.square(data) * 0
        return (np.abs(data), angle)

    def simulateIntraCurrent3D(self, angle_start_x=0.0, angle_start_y=0.0, n=10000, dt=2e-6):
        dangle = sqrt(self.Dr * 2 * dt / 5 / 10)
        dipolefield = self.efield * self.dipole
        data_x = randomAngleWalk(n*10, dangle, angle_start_x, dipolefield)
        data_y = randomAngleWalk(n*10, dangle, angle_start_y, dipolefield)
        angle_x = data_x[-1]
        angle_y = data_y[-1]
        data = np.cos(data_x[::10]) * np.cos(data_y[::10])
        # data = np.array(data)
        if self.shape < 1:
            data = self.Imin + (self.Imax - self.Imin) * np.square(data)
        elif self.shape > 1:
            data = self.Imax + (self.Imin - self.Imax) * np.square(data)
        else:
            data = self.Imax + np.square(data) * 0

        return (np.abs(data), angle_x, angle_y)

    def simulateIntraEvent(self, maxtime=0.001, dt=1e-9):
        # set time step and moving step
        offrate = np.exp(-1 * self.koff * dt)
        dx = np.sqrt(self.Df * 4 * dt)
        dangle = sqrt(self.Dr * 2 * dt)
        Ppos = np.exp(-1 * self.efield * self.charge * dx / 4.114333424e-21)
        Pneg = np.exp(self.efield * self.charge * dx / 4.114333424e-21)
        probdirection = Pneg / (Ppos + Pneg)
        maxstep = int(maxtime / dt)
        fragment = []
        s = 0
        # result
        resx = None
        resy = None
        resangle = None
        rescollision = 0
        resreaction = None
        ressuccess = 0
        reacted = -1
        while s < maxstep:
            e = s + 10000
            if e > maxstep:
                e = maxstep
            fragment.append([s, e])
            s = s + 10000

        for i in fragment:
            if resx is None:
                res = randomWalk(int(i[1] - i[0]), dx, dangle, self.dipolefield, probdirection, 0,
                                 0, 0, self.xmin, self.xmax, self.ymax, self.borderwidth, 0.001, offrate, reacted)
                resx = np.array(res[0])
                resy = np.array(res[1])
                rescollision += res[4]
                resreaction = np.array(res[3])
                resangle = np.array(res[2])
                reacted = res[6]
                if len(res[0]) < int(i[1] - i[0]):
                    ressuccess = res[5]
                    break
            else:
                x0 = resx[-1]
                y0 = resy[-1]
                angle0 = resangle[-1]
                res = randomWalk(int(i[1] - i[0]), dx, dangle, self.dipolefield, probdirection, x0, y0,
                                 angle0, self.xmin, self.xmax, self.ymax, self.borderwidth, 0.001, offrate, reacted)
                resx = np.concatenate([resx, res[0]])
                resy = np.concatenate([resy, res[1]])
                resangle = np.concatenate([resangle, res[2]])
                rescollision += res[4]
                resreaction = np.concatenate([resreaction, res[3]])
                reacted = res[6]
                if len(res[0]) < int(i[1] - i[0]):
                    ressuccess = res[5]
                    break
        print(f'collision: {rescollision}, time(us): {len(resx)*dt*1e6}')
        return np.stack((resx, resy, resangle, resreaction), axis=1), rescollision, int(ressuccess)

    def simulateBaseline(self, samplingrate=5e5, maxtime=1, rms=50e-12, unit_current='pA'):
        unitmap = {'pA': 1e12, 'nA': 1e9, 'A': 1}
        scale_current = unitmap[unit_current]
        size = int(samplingrate * maxtime)
        baseline = np.random.normal(0, rms, size) + self.baseline
        baseline *= scale_current
        return baseline

    def simulateAcquisition(self, samplingrate=5e5, maxtime=20, rms=50e-12, eventfrequency=2, dwelltime=0.2, unit_current='pA'):
        unitmap = {'pA': 1e12, 'nA': 1e9, 'A': 1}
        scale_current = unitmap[unit_current]
        t0 = 0
        dt = 1 / samplingrate
        fs_list = []
        event_list = []
        data = np.zeros(10000)
        anglestart = 0
        state = True
        while True:
            t_eventfs = np.random.exponential(1 / eventfrequency, 1)
            t_dwelltime = np.random.exponential(dwelltime, 1)
            fs_list.append(int(t_eventfs * samplingrate))
            t0 += t_eventfs
            if t0 > maxtime:
                break
            event_list.append(int(t_dwelltime * samplingrate))
            t0 += t_dwelltime
            if t0 > maxtime:
                break
        size = maxtime * samplingrate
        n = int(size / 10000)
        self.fs_list = copy.deepcopy(fs_list)
        self.event_list = copy.deepcopy(event_list)
        for i in range(n):
            data = np.random.normal(self.baseline, rms, 10000) * scale_current
            s = 0

            while s < 10000:
                if fs_list[-1] >= 10000 - s and state:
                    yield data
                    fs_list[-1] -= 10000 - s
                    s = 10000
                    if fs_list[-1] == 0:
                        fs_list.pop()
                        state = False
                elif state:
                    if fs_list[-1] == 0:
                        fs_list.pop()
                        state = False
                        continue
                    s += fs_list.pop()
                    state = False
                if s < 10000 and not state:
                    # print(event_list[-1])
                    if event_list[-1] >= 10000 - s:
                        tmp2 = self.simulateIntraCurrent(
                            anglestart, 10000 - s, dt)
                        data[s:10000] += tmp2[0] * scale_current
                        anglestart = tmp2[1]
                        yield data
                        event_list[-1] -= 10000 - s
                        s = 10000
                        if event_list[-1] == 0:
                            event_list.pop()
                            state = True
                    else:
                        if event_list[-1] == 0:
                            event_list.pop()
                            state = True
                            continue
                        tmp = event_list.pop()
                        tmp2 = self.simulateIntraCurrent(anglestart, tmp, dt)
                        data[s:s+tmp] += tmp2[0] * scale_current
                        anglestart = tmp2[1]
                        s += tmp
                        state = True
                if len(fs_list) == 0 or len(event_list) == 0:
                    return

    def simulateAcquisition3D(self, samplingrate=5e5, maxtime=20, rms=50e-12, eventfrequency=2, dwelltime=0.2, unit_current='pA'):
        unitmap = {'pA': 1e12, 'nA': 1e9, 'A': 1}
        scale_current = unitmap[unit_current]
        t0 = 0
        dt = 1 / samplingrate
        fs_list = []
        event_list = []
        data = np.zeros(10000)
        anglestartx = 0
        anglestarty = 0
        state = True
        while True:
            t_eventfs = np.random.exponential(1 / eventfrequency, 1)
            t_dwelltime = np.random.exponential(dwelltime, 1)
            fs_list.append(int(t_eventfs * samplingrate))
            t0 += t_eventfs
            if t0 > maxtime:
                break
            event_list.append(int(t_dwelltime * samplingrate))
            t0 += t_dwelltime
            if t0 > maxtime:
                break
        size = maxtime * samplingrate
        n = int(size / 10000)
        self.fs_list = copy.deepcopy(fs_list)
        self.event_list = copy.deepcopy(event_list)
        for i in range(n):
            data = np.random.normal(self.baseline, rms, 10000) * scale_current
            s = 0

            while s < 10000:
                if fs_list[-1] >= 10000 - s and state:
                    yield data
                    fs_list[-1] -= 10000 - s
                    s = 10000
                    if fs_list[-1] == 0:
                        fs_list.pop()
                        state = False
                elif state:
                    if fs_list[-1] == 0:
                        fs_list.pop()
                        state = False
                        continue
                    s += fs_list.pop()
                    state = False
                if s < 10000 and not state:
                    # print(event_list[-1])
                    if event_list[-1] >= 10000 - s:
                        tmp2 = self.simulateIntraCurrent3D(
                            anglestartx, anglestarty, 10000 - s, dt)
                        data[s:10000] += tmp2[0] * scale_current
                        anglestartx = tmp2[1]
                        anglestarty = tmp2[2]
                        yield data
                        event_list[-1] -= 10000 - s
                        s = 10000
                        if event_list[-1] == 0:
                            event_list.pop()
                            state = True
                    else:
                        if event_list[-1] == 0:
                            event_list.pop()
                            state = True
                            continue
                        tmp = event_list.pop()
                        tmp2 = self.simulateIntraCurrent3D(
                            anglestartx, anglestarty, tmp, dt)
                        data[s:s+tmp] += tmp2[0] * scale_current
                        anglestartx = tmp2[1]
                        anglestarty = tmp2[2]
                        s += tmp
                        state = True
                if len(fs_list) == 0 or len(event_list) == 0:
                    return
if __name__ == "__main__":
    nanopore = NanoporeSimulator()
    print(nanopore)
