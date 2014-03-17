import numpy
from matplotlib import pyplot
from calibrators import SmoothIso, SmoothMovingAverage, spearman, kendall

numpy.random.seed(0)
# Generate some fake data with a lognormal distribution
m = 10000
n = 10
sigma = 1.0
X = numpy.random.normal(size=(m,n))
beta = 2.0 * numpy.random.binomial(1,.5,size=n) * numpy.random.uniform()
eta = numpy.dot(X, beta)
mu = eta
y = numpy.random.lognormal(mean=mu, sigma=sigma, size=m)

# Do a linear regression on the log of the data
z = numpy.log(y)
beta_hat = numpy.linalg.lstsq(X, z)[0]
z_hat = numpy.dot(X, beta_hat)

# Try doing linear regression directly
beta_hat_direct = numpy.linalg.lstsq(X, y)[0]
y_hat = numpy.dot(X, beta_hat_direct)

# Compare the two models
rho = spearman(y, z_hat)
rho_direct = spearman(y, y_hat)
tau = kendall(y, z_hat)
tau_direct = kendall(y, y_hat)
print 'rho is %f for the log model and %f for the direct model' % (rho, rho_direct)
print 'tau is %f for the log model and %f for the direct model' % (tau, tau_direct)

# Range for plotting calibration curves
z_range = numpy.arange(z_hat.min(), z_hat.max(), .05)

# Try reversing the log by inversion
y_hat_inv = numpy.exp(z_range)

# Try reversing the log by the actual correct formula
y_hat_correct = numpy.exp(z_range + (sigma**2)/2.0)

# Try reversing the log using SmoothIso
smooth_iso = SmoothIso(max_degree=2).fit(z_hat, y)
y_hat_si = smooth_iso.predict(z_range)

# Try reversing the log using SmoothMovingAverage
moving_average = SmoothMovingAverage(max_degree=2).fit(z_hat, y)
y_hat_sma = moving_average.predict(z_range)

# Plot the different reversal attempts
lw = 2
pyplot.plot(z_hat, y, 'k.', label='data', lw=lw)
pyplot.plot(z_hat, y_hat, 'b.', label='direct regression', lw=lw)
pyplot.plot(z_range, y_hat_inv, 'r', label='$e^{\hat{z}}$', lw=lw)
pyplot.plot(z_range, y_hat_correct, 'r--', label='$e^{\hat{z} + \sigma^{2}/2 }$', lw=lw)
pyplot.plot(z_range, y_hat_si, 'y', label='smooth iso', lw=lw)
pyplot.plot(z_range, y_hat_sma, 'g--', label='smooth moving average', lw=lw)
pyplot.ylim(0,40)
pyplot.legend(loc=0)
pyplot.xlabel('log-scale prediction')
pyplot.ylabel('original scale')
pyplot.savefig('example.png', transparent=True)
pyplot.show()



