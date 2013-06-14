import numpy
import matplotlib.pyplot as pyplot
from scipy.optimize import curve_fit
from scipy.stats import norm

numpy.random.seed(100)

mass = numpy.arange(100, 160, 2)
background = 28239.0 * numpy.exp(-0.0209 * mass); 
background = numpy.sqrt(background) * numpy.random.randn(mass.shape[0]) + background
signal = numpy.exp(-(mass-125.0)**2 * 0.07)*150
signal = numpy.sqrt(signal) * numpy.random.randn(mass.shape[0]) + signal
data = background + signal

def fit_fn(m, A_bkg, k_bkg, A_sig, avg_sig, k_sig):
  return A_bkg*numpy.exp(-k_bkg*m) + A_sig*numpy.exp(-k_sig*(m-avg_sig)**2)
def bkg_fn(m, A_bkg, k_bkg, A_sig, avg_sig, k_sig):
  return A_bkg*numpy.exp(-k_bkg*m)

p0 = (40000, 0.015, 170, 125, 0.1)
popt, pcov = curve_fit(fit_fn, mass, data, sigma = numpy.sqrt(data), p0=p0)
print popt

#pyplot.plot(mass, background, '.')
pyplot.errorbar(mass, data, yerr=numpy.sqrt(data), fmt='.')
pyplot.ylim(0, 4000)
pyplot.ylabel("Events / 2 GeV")
pyplot.xlabel("m_yy [GeV]")
pyplot.savefig("data1.png")
pyplot.plot(mass, fit_fn(mass, *popt))
pyplot.plot(mass, bkg_fn(mass, *popt))
pyplot.legend(("Data", "Sig + Bkg fit", "Bkg fit"), 'upper right', fancybox=True)
pyplot.savefig("data2.png")
pyplot.clf()

pyplot.errorbar(mass, data - bkg_fn(mass, *popt), yerr=numpy.sqrt(data), fmt='.')
pyplot.plot(mass, fit_fn(mass, *popt) - bkg_fn(mass, *popt))
pyplot.ylabel("Events / 2 GeV")
pyplot.xlabel("m_yy [GeV]")
pyplot.legend(("Data - Bkg fit", "Sig fit"), 'upper right', fancybox=True)
pyplot.savefig("bkgsub.png")
pyplot.clf()

zscores = (data - bkg_fn(mass, *popt)) / numpy.sqrt(data)
pvalues = norm.sf(zscores)
pyplot.plot(mass, pvalues)
pyplot.ylabel("P value")
pyplot.xlabel("m_yy [GeV]")
pyplot.yscale("log")
pyplot.savefig("pval.png")

