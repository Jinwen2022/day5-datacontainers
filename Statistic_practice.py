from scipy.stats import poisson
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm

#a

fig, ax = plt.subplots(1, 1)

mu = 0.6
mean, var, skew, kurt = poisson.stats(mu, moments='mvsk')
x = np.arange(poisson.ppf(0.01, mu),
              poisson.ppf(0.99, mu))
print(x)
ax.plot(x, poisson.pmf(x, mu), 'bo', ms=8, label='poisson pmf')
ax.vlines(x, 0, poisson.pmf(x, mu), colors='b', lw=5, alpha=0.5)
rv = poisson(mu)
ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,
        label='frozen pmf')
ax.legend(loc='best', frameon=False)
plt.show()

fig2, ax = plt.subplots(1, 1)

prob = poisson.cdf(x, mu)
np.allclose(x, poisson.ppf(prob, mu))
ax.plot(x, poisson.cdf(x, mu), 'bo', ms=8, label='poisson cdf')
ax.vlines(x, 0, poisson.cdf(x, mu), colors='b', lw=5, alpha=0.5)
rv = poisson(mu)
ax.vlines(x, 0, rv.cdf(x), colors='k', linestyles='-', lw=1,
        label='frozen cdf')
ax.legend(loc='best', frameon=False)
plt.show()

r1 = poisson.rvs(mu, size=1000)
plt.hist(r1)
plt.show()

#Create a continious random variable with normal distribution and plot its probability mass function (PMF),
# cummulative distribution function (CDF) and a histogram of 1000 random realizations of the variable

fig, ax = plt.subplots(1, 1)
mean, var, skew, kurt = norm.stats(moments='mvsk')
x = np.linspace(norm.ppf(0.01),
                norm.ppf(0.99), 100)
ax.plot(x, norm.pdf(x),
       'r-', lw=5, alpha=0.6, label='norm pdf')
rv = norm()
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
vals = norm.ppf([0.001, 0.5, 0.999])
np.allclose([0.001, 0.5, 0.999], norm.cdf(vals))
r2 = norm.rvs(size=1000)
ax.hist(r2, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()


###c. Test if two sets of (independent) random data comes from the same distribution
print(stats.ttest_ind(r1, r2, trim=.2), 'so it is not')