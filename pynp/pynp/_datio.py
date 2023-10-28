from .cfunction import Downsampler 
from matplotlib.axes import Axes 


class DatSampler(object):
    def __init__(self, fn: str or np.ndarray or None, fs = 500, ax :Axes = None) -> None:
        super().__init__()
        self.downsampler = Downsampler(fn, fs)
        self.fn = fn 
        self.line = None
        if ax is not None:
            ax.callbacks.connect('xlim_changed', self.update)
            ax.plot(*self.downsampler.downsample(0, -1))
            self.line = ax.lines[-1]

    def setAxes(self, ax : Axes):
        ax.callbacks.connect('xlim_changed', self.update)
        ax.plot(*self.downsampler.downsample(0, -1))
        self.line = ax.lines[-1]

    def update(self, ax:Axes):
        xstart, xend = ax.get_xlim()
        self.line.set_data(*self.downsampler.downsample(xstart, xend))
        ax.figure.canvas.draw_idle()




