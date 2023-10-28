from .._porephysics import Nanopore

def calculateMV(physical = {"radius":5e-9, "length":27e-9, "resistivity":0.046, "voltage":-1}, parameter = [10000, 12000, 50000]):
    n=Nanopore(physical)
    print(n(*parameter))