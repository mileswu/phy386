#/usr/bin/env python

import numpy
import math
import matplotlib.pyplot as pyplot

n = 1024
deltat = 1.0

t = numpy.arange(n) * deltat
whitenoise = numpy.random.normal(0.0, 1.0, n)
sinewave = 0.4*numpy.sin(t * 2.0 * math.pi / n / deltat)

pyplot.plot(t, whitenoise + sinewave)
pyplot.show()

whitenoise_fft = numpy.fft.rfft(whitenoise + sinewave)
freq = numpy.arange(n/2 + 1) / deltat / n

freqfilter = numpy.append(numpy.ones(freq.size/10), numpy.zeros(freq.size - freq.size/10))
whitenoise_fft_filtered = whitenoise_fft * freqfilter

#Integral of power
whitenoise_fft_filtered_power = numpy.abs(whitenoise_fft_filtered/n)**2 * 2.0
print freq[1]
print numpy.sum(whitenoise_fft_filtered_power)*freq[1]
pyplot.plot(freq, whitenoise_fft_filtered_power)

whitenoise_filtered = numpy.fft.irfft(whitenoise_fft_filtered)
autocorr = numpy.correlate(whitenoise_filtered, whitenoise_filtered, 'full')[n-1:]
autocorr = autocorr / numpy.arange(n, 0, -1)

print numpy.sum(autocorr)*t[1]


#pyplot.plot(t, whitenoise_filtered)
pyplot.plot(t, autocorr)
pyplot.show()
