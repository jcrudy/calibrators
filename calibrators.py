from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.isotonic import IsotonicRegression
import numpy
from pyearth.earth import Earth
from scipy.interpolate.fitpack2 import UnivariateSpline
from moving_average import moving_average

class SmoothMovingAverage(BaseEstimator, RegressorMixin):
    def __init__(self, window_size=None):
        self.window_size = window_size
    
    def fit(self, X, y):
        if self.window_size is None:
            window_size = len(X) / 10
        else:
            window_size = self.window_size
            
        order = numpy.argsort(X)
        y_ = moving_average(y[order], window_size)
        x_ = X[order][int(window_size)/2 - 1:-int(window_size)/2]
        self.spline_ = Earth(max_degree=1, smooth=True).fit(x_, y_)
    
    def predict(self, X):
        return self.spline_.predict(X)
    
    def transform(self, X):
        return self.predict(X)

class SmoothIso(BaseEstimator, RegressorMixin):
    def __init__(self, y_min=None, y_max=None, max_degree=None):
        self.y_min = y_min
        self.y_max = y_max
        
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
        self.spline_ = Earth(max_degree=2, smooth=True).fit(self.X_, self.y_, sample_weight=self.w_)
        print self.spline_.summary()
        
    def predict(self, X):
        return self.spline_.predict(X)
    
    def transform(self, X):
        return self.predict(X)
