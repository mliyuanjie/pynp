import matplotlib.pyplot as plt
import matplotlib.patches as patches 
from matplotlib.axes import Axes 
import matplotlib.lines as lines
import numpy as np
import time 
from math import pi
from io import BytesIO 
from PIL import Image
from tkinter import filedialog
import tkinter as tk

def _trajDisplayer(fn, ax: Axes, dt = 1e-9, frame = 0.02, speed = 1, porelength = 30e-9, poreradius = 10e-9, radiusx = 3e-9, radiusy = 1.8e-9):
    data = np.fromfile(fn, 'float32').reshape(4,-1)
    fig = ax.figure
    ax.set_axis_off() 
    dt *= 1e9 
    porelength *= 1e9
    poreradius *= 1e9
    radiusx *= 1e9
    radiusy *= 1e9
    ax.add_patch(patches.Rectangle((-1.5 * poreradius, 0), 0.5 * poreradius, porelength, edgecolor = "orange", facecolor = "gray", fill = True, lw = 5))
    ax.add_patch(patches.Rectangle((poreradius, 0), 0.5 * poreradius, porelength, edgecolor = "orange", facecolor = "gray", fill = True, lw = 5))
    protein = patches.Ellipse((0,0), 2 * radiusx, 2 * radiusy, color = 'blue', clip_on = False) 
    line = lines.Line2D([], [], linewidth = 0.5, color ='black')
    ax.set_xlim(-1.5 * poreradius, 1.5 * poreradius)
    ax.set_ylim(-1 * poreradius, porelength + poreradius)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    
    bg = fig.canvas.copy_from_bbox(fig.bbox)


    ax.add_patch(protein) 
    ax.add_line(line)
    ipre = 0
    jpre = 0
    passtime = 0
    timetext = ax.text(poreradius, porelength + 5, f'{0} ns')
    count = 1
    for num in range(data[0].size):
        if count < speed:
            count+=1
            continue
        count = 1
        i = data[0][num]
        j = data[1][num]
        theta = data[2][num] * 180 / pi 
        state = data[3][num]
        #update trajectory plot
        fig.canvas.restore_region(bg)
        #if k > 0:
        #    protein.set_color('red')
        #else:
        line.set_data([ipre, i], [jpre, j])
        ipre = i
        jpre = j
        ax.draw_artist(line)
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()
        bg = fig.canvas.copy_from_bbox(fig.bbox)
        #update protein position 
        #fig.canvas.restore_region(bg)
        passtime += dt
        if passtime > 1000:
            timetext.set_text(f'{passtime / 1000} us')
        elif passtime > 1e6:
            timetext.set_text(f'{passtime / 1e6} ms')
        else:
            timetext.set_text(f'{passtime} ns')
        #protein.set_visible(True)
        #timetext.set_visible(True)
        if state > 0:
            protein.set_color('red')
        else:
            protein.set_color('blue')
        protein.center = i, j
        protein.angle = theta
        ax.draw_artist(protein)
        ax.draw_artist(timetext)
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()
        time.sleep(frame)
        #protein.set_visible(False)
        #timetext.set_visible(False)
        
def _trajDisplayer_save(fn, ax: Axes, dt = 1e-9, frame = 0.02, speed = 1, porelength = 30e-9, poreradius = 10e-9, radiusx = 3e-9, radiusy = 1.8e-9):
    data = np.fromfile(fn, 'float32').reshape(4,-1)
    fig = ax.figure
    ax.set_axis_off() 
    dt *= 1e9 
    porelength *= 1e9
    poreradius *= 1e9
    radiusx *= 1e9
    radiusy *= 1e9
    ax.add_patch(patches.Rectangle((-1.5 * poreradius, 0), 0.5 * poreradius, porelength, edgecolor = "orange", facecolor = "gray", fill = True, lw = 5))
    ax.add_patch(patches.Rectangle((poreradius, 0), 0.5 * poreradius, porelength, edgecolor = "orange", facecolor = "gray", fill = True, lw = 5))
    protein = patches.Ellipse((0,0), 2 * radiusx, 2 * radiusy, color = 'blue', clip_on = False) 
    line = lines.Line2D([], [], linewidth = 0.5, color ='black')
    ax.set_xlim(-1.5 * poreradius, 1.5 * poreradius)
    ax.set_ylim(-1 * poreradius, porelength + poreradius)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    
    bg = fig.canvas.copy_from_bbox(fig.bbox)


    ax.add_patch(protein) 
    ax.add_line(line)
    ipre = 0
    jpre = 0
    passtime = 0
    timetext = ax.text(poreradius, porelength + 5, f'{0} ns')
    count = 1
    imlist = []
    for num in range(data[0].size):
        if count < speed:
            count+=1
            continue
        count = 1
        i = data[0][num]
        j = data[1][num]
        theta = data[2][num] * 180 / pi 
        state = data[3][num]
        #update trajectory plot
        fig.canvas.restore_region(bg)
        #if k > 0:
        #    protein.set_color('red')
        #else:
        line.set_data([ipre, i], [jpre, j])
        ipre = i
        jpre = j
        ax.draw_artist(line)
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()
        bg = fig.canvas.copy_from_bbox(fig.bbox)
        #update protein position 
        #fig.canvas.restore_region(bg)
        passtime += dt*speed
        if passtime > 1000:
            timetext.set_text(f'{passtime / 1000} us')
        elif passtime > 1e6:
            timetext.set_text(f'{passtime / 1e6} ms')
        else:
            timetext.set_text(f'{passtime} ns')
        #protein.set_visible(True)
        #timetext.set_visible(True)
        if state > 0:
            protein.set_color('red')
        else:
            protein.set_color('blue')
        protein.center = i, j
        protein.angle = theta
        ax.draw_artist(protein)
        ax.draw_artist(timetext)
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()
        time.sleep(frame)
        buf = BytesIO()
        plt.savefig(buf)
        buf.seek(0)
        im = Image.open(buf)
        imlist.append(im)
    imlist[0].save(fn[:-4]+'gif',save_all = True, append_images=imlist[1:])

def trajDisplay(dt = 1e-9, frame = 0.02, speed = 10000, porelength = 30e-9, poreradius = 10e-9, radius = 3e-9):
    root = tk.Tk()
    file_path = filedialog.askopenfilename()
    root.destroy()

    fig = plt.figure(figsize=(5, 5*1.667)) 
    fig.show()
    ax = fig.add_axes([0,0,1,1])
    time.sleep(0)
    _trajDisplayer_save(file_path, ax, dt, frame, speed, porelength, poreradius, radius)
    

    

if __name__ == "__main__":
    trajDisplay()





