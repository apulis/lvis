import numpy as np


def linear(step, total, **kwargs): 
    self_phase = kwargs.get('phase', None)
    self_eps = kwargs.get('eps', 0)
    phase = step/total

    if self_phase and phase >= self_phase:
        return self_eps
    return 1.0-phase + self_eps

def constant(step, total, **kwargs):
    self_phase = kwargs.get('phase', None)
    self_eps = kwargs.get('eps', 0)
    phase = step/total

    if self_phase and phase >= self_phase:
        return self_eps
    return 1.0

def convex(step, total, **kwargs): 
    self_phase = kwargs.get('phase', None)
    self_eps = kwargs.get('eps', 0)
    phase = step/total

    if self_phase and phase >= self_phase:
        return self_eps
    return np.cos(phase*np.pi/2) + self_eps

def concave(step, total, **kwargs):
    pass

def composite(step, total, **kwargs):
    self_phase = kwargs.get('phase', None)
    self_eps = kwargs.get('eps', 0)
    phase = step/total

    if self_phase and phase >= self_phase:
        return self_eps
    return 0.5*(1+np.cos(phase*np.pi)) + self_eps

def quadratic(step, total, **kwargs):
    self_phase = kwargs.get('phase', None)
    self_eps = kwargs.get('eps', 0)
    phase = step/total

    if self_phase and phase >= self_phase:
        return self_eps
    return (1-phase)**2 + self_eps