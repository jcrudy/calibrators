import numpy
from matplotlib import pyplot
import math

def moving_average(y, window_size=10):
    window = numpy.ones(int(window_size)) / float(window_size)
    return numpy.convolve(y, window, 'valid')

class RunningMean(object):
    def __init__(self):
        self.mean = 0.0
        self.n = 0.0
        
    def eval(self):
        return self.mean
    
    def add(self, val):
        self.mean = (self.n / float(self.n+1))*self.mean + (1.0/float(self.n+1))*val
        self.n += 1.0
        
    def remove(self, val):
        self.n -= 1.0
        self.mean = (self.mean - (1.0/(self.n+1))*val) / (self.n / (self.n+1))
        
class RunningStandardDeviation(object):
    def __init__(self):
        self.s0 = 0.0
        self.s1 = 0.0
        self.s2 = 0.0
        
    def eval(self):
        return math.sqrt((self.s0*self.s2 - self.s1*self.s1) / (self.s0*(self.s0 - 1)))
    
    def add(self, val):
        self.s0 += 1.0
        self.s1 += val
        self.s2 += val*val
        
    def remove(self, val):
        self.s0 -= 1.0
        self.s1 -= val
        self.s2 -= val*val

def adaptive_moving_average(y, max_window=None):
    '''
    Assumes y is sorted
    '''
    if max_window is None:
        max_window = len(y) / 100
    weights = numpy.empty_like(y)
    means = numpy.empty_like(y)
    len_y = len(y)
    n = 0
    i_low = 0
    i = 0
    i_high = 0
    mn = 0
    while True:
        if i >= len_y:
            break
        if n >= max_window:
            mn -= y[i_low]
            i_low += 1
            n -= 1
        mn += y[i_high]
        i_high += 1
        n += 1
        means[i] = mn
        weights[i] = n
        i += 1
    return means, weights
    
def moving_average_plot(x, y, window_size=10, *args, **kwargs):
    order = numpy.argsort(x)
    y_ = moving_average(y[order], window_size)
    pyplot.plot(x[order][int(window_size)/2 - 1:-int(window_size)/2], y_, *args, **kwargs)
    
