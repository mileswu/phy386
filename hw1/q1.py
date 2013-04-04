#/usr/bin/env python

import numpy
import math
import matplotlib.pyplot as pyplot

n = 1024
deltat = 0.01
amplitude = 1.0

t = numpy.arange(n) * deltat
sinewave = amplitude * numpy.sin(t * 2.0 * math.pi / n / deltat)

#pyplot.plot(t, sinewave)
#pyplot.show()

sinewave_fft = numpy.fft.rfft(sinewave)
freq = numpy.arange(n/2 + 1) / deltat / n

sinewave_fft_absrmsnormalized = numpy.abs(sinewave_fft) / n * math.sqrt(2.0)

pyplot.plot(freq, numpy.abs(sinewave_fft_absrmsnormalized))
pyplot.show()
