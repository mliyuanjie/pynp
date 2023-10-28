import h5py 
import numpy as np 
import pandas as pd 
from tkinter import filedialog
    


def convertHDF5():
    fn_dat = filedialog.askopenfilename(filetypes=[("event file", "*.dat")])
    fn_h5 = fn_dat[:-3] + 'hdf5'
    fn_csv = fn_dat[:-3] + 'csv'
    f = h5py.File(fn_h5, 'w')
    f.create_group("eventdata")
    f.create_group("eventlist")

    data = np.memmap(fn_dat, dtype = 'float32', mode = 'r')
    pf = pd.read_csv(fn_csv, index_col = None)
    eventlist = pf.to_dict(orient = 'list') 

    for k, v in eventlist.items():
        f.create_dataset('/eventlist/' + k, data = np.array(v))    
    
    for i in range(len(eventlist['start'])):
        f.create_dataset('/eventdata/'+str(i), data = data[eventlist['begin'][i]:eventlist['end'][i]]) 
    f.close()
