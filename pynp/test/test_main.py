import cfunction
import numpy as np 

#a= cfunction.__name__() 

a = np.array([-100.0, -102, -100, -80, -100, -102, -100, -100, -100])
b = np.flip(a)

print(cfunction.test(b))