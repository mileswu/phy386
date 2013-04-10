#/usr/bin/env python

import numpy
import math
import matplotlib.pyplot as pyplot

n = 1024
deltat = 1.0/n
amplitude = 1.0

t = numpy.arange(n) * deltat
sinewave = amplitude * numpy.sin(t * 2.0 * math.pi / n / deltat)

sinewave_fft = numpy.fft.fft(sinewave)
sinewave_fft2 = numpy.append(sinewave_fft[n/2:], sinewave_fft[:n/2])
freq = numpy.arange(-n/2, n/2) / deltat / n
sinewave_fft_absrmsnormalized = numpy.abs(sinewave_fft2) / n

pyplot.figure(1, figsize=(6.0, 7.0))

pyplot.subplot(211)
pyplot.plot(t, sinewave)
pyplot.xlabel("Time [s]")
pyplot.ylabel("Signal [arb]")
pyplot.subplots_adjust(hspace = 0.8)

pyplot.subplot(212)
pyplot.plot(freq[n/2-10:n/2+10], sinewave_fft_absrmsnormalized[n/2-10:n/2+10])
pyplot.xlabel("Frequency [Hz]")
pyplot.ylabel("Power spectrum [arb]")

pyplot.savefig("q1.png")
#pyplot.show()
