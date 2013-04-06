#/usr/bin/env python

import numpy
import math
import matplotlib.pyplot as pyplot
import pyfits

spt_fits = pyfits.open("/Users/mileswu/spt/field_scan_150_20080523_145759.fits")
spt_adc = spt_fits[9].data.field("ADC")[0]

spt_adc_to_watts = spt_fits[2].data.field("BOLO_ADC_TO_WATTS")[0] #this is per channel
#spt_adc = spt_adc[numpy.array([1])]
#spt_adc_to_watts = spt_adc_to_watts[numpy.array([1])]
spt_watts = spt_adc * spt_adc_to_watts[:, numpy.newaxis]
spt_watts = spt_watts[numpy.where(spt_adc_to_watts != 0)]

spt_scanmask = spt_fits[4].data.field("SCAN_FLAG_REDUCED")[0]
splitpoints_start = numpy.where(numpy.diff(spt_scanmask) > 0)[0] + 1
if spt_scanmask[0] == 1:
	numpy.insert(splitpoints_start, 0, 0)

splitpoints_end = numpy.where(numpy.diff(spt_scanmask) < 0)[0] + 1
if spt_scanmask[-1] == 1:
	numpy.insert(splitpoints_end, -1, spt_scanmask.size)

splitpoints_minlength = numpy.amin(splitpoints_end - splitpoints_start)

fft_len = int(2**math.floor(math.log(splitpoints_minlength,2.0)))
fft_hanning = numpy.hanning(fft_len)

spt_watts_splitbytime = numpy.array([spt_watts[:, i:i+fft_len] for i in splitpoints_start])
#skipping subtracting off average
spt_fft_splitbytime = numpy.fft.rfft(spt_watts_splitbytime * fft_hanning)


spt_fft_chan = spt_fft_splitbytime[:,0,:] * numpy.conjugate(spt_fft_splitbytime)[:,1,:]
spt_fft_chan = numpy.sum(spt_fft_chan, 0)
spt_fft_chan = spt_fft_chan * (2/(100*numpy.sum(fft_hanning**2))) / splitpoints_start.size

delta_t = 1.0/100
fft_freqs = numpy.arange(fft_len/2 + 1) / delta_t / fft_len

pyplot.plot(fft_freqs[2:], numpy.sqrt(numpy.abs(spt_fft_chan))[2:])
pyplot.yscale("log")
pyplot.xscale("log")
