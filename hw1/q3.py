#/usr/bin/env python

import numpy
import math
import matplotlib.pyplot as pyplot
import pyfits

delta_t = 1.0/100.0

spt_fits = pyfits.open("/Users/mileswu/spt/field_scan_150_20080523_145759.fits")
spt_adc = spt_fits[9].data.field("ADC")[0]

spt_adc_to_watts = spt_fits[2].data.field("BOLO_ADC_TO_WATTS")[0] #this is per channel
print "Available channels"
print numpy.where(spt_adc_to_watts != 0)

print "Picking channel 1 and 2"
spt_adc = spt_adc[numpy.array([1,2])]
spt_adc_to_watts = spt_adc_to_watts[numpy.array([1,2])]

spt_watts = spt_adc * spt_adc_to_watts[:, numpy.newaxis]
print "Done SPT watts conversion"

spt_scanmask = spt_fits[4].data.field("SCAN_FLAG_REDUCED")[0]
splitpoints_start = numpy.where(numpy.diff(spt_scanmask) > 0)[0] + 1
if spt_scanmask[0] == 1:
	numpy.insert(splitpoints_start, 0, 0)
splitpoints_end = numpy.where(numpy.diff(spt_scanmask) < 0)[0] + 1
if spt_scanmask[-1] == 1:
	numpy.insert(splitpoints_end, -1, spt_scanmask.size)
splitpoints_minlength = numpy.amin(splitpoints_end - splitpoints_start)

# plotting for Q3a
pyplot.figure(1, figsize=(6.0, 7.0))
pyplot.subplot(311)
pyplot.plot(numpy.arange(0.0, 10.0, delta_t), spt_watts[0][:10.0/delta_t])
pyplot.xlabel("Time [s]")
pyplot.ylabel("Power Ch 1 [W]")
pyplot.subplots_adjust(hspace = 0.8)

pyplot.subplot(312)
pyplot.plot(numpy.arange(0.0, 60.0, delta_t), spt_watts[0][:60.0/delta_t])
pyplot.xlabel("Time [s]")
pyplot.ylabel("Power Ch 1 [W]")
pyplot.subplots_adjust(hspace = 0.8)

pyplot.subplot(313)
pyplot.plot(numpy.arange(0.0, 600.0, delta_t), spt_watts[0][:600.0/delta_t])
pyplot.xlabel("Time [s]")
pyplot.ylabel("Power Ch 1 [W]")
pyplot.subplots_adjust(hspace = 0.8)

pyplot.savefig("q3a.png")
pyplot.clf()

#plotting for Q3b
pyplot.figure(1, figsize=(6.0, 7.0))
pyplot.subplot(311)
pyplot.plot(numpy.arange(0.0, 10.0, delta_t), spt_watts[0][:10.0/delta_t])
pyplot.plot(numpy.arange(0.0, 10.0, delta_t), spt_watts[1][:10.0/delta_t])
pyplot.xlabel("Time [s]")
pyplot.ylabel("Power Ch 1/2 [W]")
pyplot.subplots_adjust(hspace = 0.8)

pyplot.subplot(312)
pyplot.plot(numpy.arange(0.0, 60.0, delta_t), spt_watts[0][:60.0/delta_t])
pyplot.plot(numpy.arange(0.0, 60.0, delta_t), spt_watts[1][:60.0/delta_t])
pyplot.xlabel("Time [s]")
pyplot.ylabel("Power Ch 1/2 [W]")
pyplot.subplots_adjust(hspace = 0.8)

pyplot.subplot(313)
pyplot.plot(numpy.arange(0.0, 600.0, delta_t), spt_watts[0][:600.0/delta_t])
pyplot.plot(numpy.arange(0.0, 600.0, delta_t), spt_watts[1][:600.0/delta_t])
pyplot.xlabel("Time [s]")
pyplot.ylabel("Power Ch 1/2 [W]")
pyplot.subplots_adjust(hspace = 0.8)

pyplot.savefig("q3b.png")
pyplot.clf()


# Q4
fft_len = int(2**math.floor(math.log(splitpoints_minlength,2.0)))
fft_hanning = numpy.hanning(fft_len)

spt_watts_splitbytime = numpy.array([spt_watts[:, i:i+fft_len] for i in splitpoints_start])
print "Split by time"

spt_fft_splitbytime = numpy.fft.rfft(spt_watts_splitbytime * fft_hanning)
fft_freqs = numpy.arange(fft_len/2 + 1) / delta_t / fft_len
print "Done FFT"

spt_fft_chan0 = spt_fft_splitbytime[:,0,:] * numpy.conjugate(spt_fft_splitbytime)[:,0,:]
spt_fft_chan0 = numpy.sum(spt_fft_chan0, 0)
spt_fft_chan0 = spt_fft_chan0 * 2 * delta_t / numpy.sum(fft_hanning**2) / splitpoints_start.size

spt_fft_chan01 = spt_fft_splitbytime[:,0,:] * numpy.conjugate(spt_fft_splitbytime)[:,1,:]
spt_fft_chan01 = numpy.sum(spt_fft_chan01, 0)
spt_fft_chan01 = spt_fft_chan01 * 2 * delta_t / numpy.sum(fft_hanning**2) / splitpoints_start.size

pyplot.figure(2, figsize=(6.0, 12.0))

pyplot.subplot(311)
pyplot.plot(fft_freqs[2:], numpy.sqrt(numpy.abs(spt_fft_chan0))[2:])
pyplot.xlabel("Frequency [Hz]")
pyplot.ylabel("Power spectral density (ch1) [W/sqrt(Hz)]")
pyplot.yscale("log")
pyplot.xscale("log")
pyplot.subplots_adjust(hspace = 0.5)

pyplot.subplot(312)
pyplot.plot(fft_freqs[2:], numpy.sqrt(numpy.abs(spt_fft_chan01))[2:])
pyplot.xlabel("Frequency [Hz]")
pyplot.ylabel("Cross magnitude (ch1/2) [W/sqrt(Hz)]")
pyplot.yscale("log")
pyplot.xscale("log")

pyplot.subplot(313)
pyplot.plot(fft_freqs[2:], numpy.angle(spt_fft_chan01)[2:])
pyplot.xlabel("Frequency [Hz]")
pyplot.ylabel("Cross phase (ch1/2) [rad]")
pyplot.xscale("log")

pyplot.savefig("q4.png")
