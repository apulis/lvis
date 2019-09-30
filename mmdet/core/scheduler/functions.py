import numpy as np


def linear(phase): 
    return 1.0-phase 

def constant(phase):
    return 1.0

def convex(phase): 
    return np.cos(phase*np.pi/2)

def concave(phase):
    pass

def composite(phase):
    return 0.5*(1+np.cos(phase*np.pi))