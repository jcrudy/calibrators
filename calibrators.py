from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.isotonic import IsotonicRegression
import numpy
from pyearth.earth import Earth
from matplotlib import pyplot
import pandas

def moving_average(y, window_size):
    window = numpy.ones(int(window_size)) / float(window_size)
    return numpy.convolve(y, window, 'valid')

def moving_average_plot(x, y, window_size, *args, **kwargs):
    order = numpy.argsort(x)
    y_ = moving_average(y[order], window_size)
    pyplot.plot(x[order][int(window_size)/2 - 1:-int(window_size)/2], y_, *args, **kwargs)
    
def kendall(observed, predicted):
    return pandas.Series(observed).corr(pandas.Series(predicted), method='kendall')

def spearman(observed, predicted):
    return pandas.Series(observed).corr(pandas.Series(predicted), method='spearman')

class SmoothMovingAverage(BaseEstimator, RegressorMixin):
    def __init__(self, window_size=None, **kwargs):
        self.window_size = window_size
        self.kwargs = kwargs
    
    def fit(self, X, y):
        if self.window_size is None:
            window_size = len(X) / 100
        else:
            window_size = self.window_size
            
        order = numpy.argsort(X)
        y_ = moving_average(y[order], window_size)
        x_ = X[order][int(window_size)/2 - 1:-int(window_size)/2]
        self.spline_ = Earth(**self.kwargs).fit(x_, y_)
        return self
    
    def predict(self, X):
        return self.spline_.predict(X)
    
    def transform(self, X):
        return self.predict(X)

class SmoothIso(BaseEstimator, RegressorMixin):
    def __init__(self, y_min=None, y_max=None, **kwargs):
        self.y_min = y_min
        self.y_max = y_max
        self.kwargs = kwargs
        
    def fit(self, X, y):
        self.iso_ = IsotonicRegression(y_min=self.y_min, y_max=self.y_max).fit(X,y)
        n = self.iso_.X_.shape[0]
        last = self.iso_.y_[0]
        current_sum = 0.0
        current_count = 0
        i = 0
        X_ = []
        y_ = []
        w_ = []
        while True:
            current = self.iso_.y_[i]
            if current != last:
                X_.append(current_sum / float(current_count))
                y_.append(last)
                w_.append(float(current_count))
                current_sum = 0.0
                current_count = 0
                last = current
            current_sum += self.iso_.X_[i]
            current_count += 1
            i += 1
            if i >= n:
                break
        self.X_ = numpy.array(X_)
        self.y_ = numpy.array(y_)
        self.w_ = numpy.array(w_)
        self.spline_ = Earth(**self.kwargs).fit(self.X_, self.y_, sample_weight=self.w_)
        return self
    
    def predict(self, X):
        return self.spline_.predict(X)
    
    def transform(self, X):
        return self.predict(X)
