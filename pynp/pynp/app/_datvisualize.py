from tkinter import filedialog
import tkinter as tk
import matplotlib.pyplot as plt
import pandas as pd 
from os import path
from .._datio import DatSampler 



def datDisplay(fs = 500):
    interval = 1 / 500
    plt.plot()
    ax = plt.gca()
    root = tk.Tk()
    file_path = filedialog.askopenfilename()
    root.destroy()
    dd = DatSampler(file_path, fs, ax)
    csv_path = file_path[:-3]+"csv" 
    if not path.exists(csv_path):
        plt.show()
        return
    df = pd.read_csv(csv_path, index_col=None)
    p = df.to_dict(orient="list")
    yminh = []
    ymaxh = []
    xminh = []
    xmaxh = []
    x = []
    y=[]
    for i in range(len(p["start(ms)"])):
        x.append(p["start(ms)"][i])
        x.append(p["start(ms)"][i])
        x.append(p["end(ms)"][i])
        x.append(p["end(ms)"][i])
        y.append(p["I0(pA)"][i])
        y.append(p["I1(pA)"][i])
        y.append(p["I1(pA)"][i])
        y.append(p["I0(pA)"][i])
            
        if "Imin" in p:
            yminh.append(p["I0(pA)"][i]+p["Imin"][i])
            ymaxh.append(p["I0(pA)"][i]+p["Imax"][i])
            xminh.append(p["begin"][i] * interval)
            xmaxh.append(p["end(ms)"][i])
            ax.hlines(yminh, xminh, xmaxh, colors = "blueviolet")
            ax.hlines(ymaxh, xminh, xmaxh, colors = "blueviolet")
        ax.plot(x, y, color = 'r')
    plt.show()