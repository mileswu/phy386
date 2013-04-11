#/usr/bin/env python

import numpy
import math
import matplotlib.pyplot as pyplot

n = 1024
deltat = 1.0

t = numpy.arange(n) * deltat
whitenoise = numpy.random.normal(0.0, 1.0, n/2+1) + 1j*numpy.random.normal(0.0, 1.0, n/2+1)
whitenoise[0] = 0
freq = numpy.arange(n/2 + 1) / deltat / n

tau = 5.0
rcfilter = 1.0 / (1.0 + 1j*2.0*math.pi*freq*tau)
whitenoise_filtered = whitenoise * rcfilter
print "Integral of PS: %f" % numpy.sum(numpy.abs(whitenoise_filtered)**2/n/n)

whitenoise_t = numpy.fft.irfft(whitenoise_filtered)

autocorr2 = numpy.correlate(whitenoise_t, whitenoise_t, 'full')[n-1:]
print "Integral of autocorrelation = %f" % numpy.sum(autocorr2/n)

pyplot.figure(1, figsize=(10.0, 12.0))

pyplot.subplot(411)
pyplot.plot(freq, numpy.abs(whitenoise)**2/n/n)
pyplot.xlabel("Frequency [Hz]")
pyplot.ylabel("Power spectrum [arb^2]")
#pyplot.subplots_adjust(hspace = 0.5)

pyplot.subplot(412)
pyplot.plot(freq, numpy.abs(whitenoise_filtered)**2/n/n)
pyplot.xlabel("Frequency [Hz]")
pyplot.ylabel("Filtered Power spectrum [arb^2]")

pyplot.subplot(413)
pyplot.plot(t, whitenoise_t)
pyplot.xlabel("Time [s]")
pyplot.ylabel("Band-limited time stream [arb]")

pyplot.subplot(414)
pyplot.plot(t, autocorr2/n)
pyplot.xlabel("Lag Time [s]")
pyplot.ylabel("Autocorrelation [arb^2]")
pyplot.savefig("q2.png")
