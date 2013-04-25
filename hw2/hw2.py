#/usr/bin/env python

import numpy
import math
import matplotlib.pyplot as pyplot
import pyfits

spt_fits = pyfits.open("fs.fits")
spt_adc = spt_fits[9].data.field("ADC")[0]

delta_t = 1.0/100.0
t = numpy.arange(0, spt_adc.shape[1]) * delta_t

spt_adc_to_watts = spt_fits[2].data.field("BOLO_ADC_TO_WATTS")[0] #this is per channel
print "Available channels"
available_channels =  numpy.where(spt_adc_to_watts != 0)[0]
print available_channels

print "Picking channels"
channels_to_pick = [1,2,3,4,5,6,7,11,12,13,14,15,16,17,18,19] # 8,20 bad
#channels_to_pick = [1,2]
#channels_to_pick = [1,2,3,4,5,6]
spt_adc = spt_adc[channels_to_pick]
spt_adc_to_watts = spt_adc_to_watts[channels_to_pick]

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

spt_watts_masked = spt_watts[:, numpy.where(spt_scanmask == 1)[0]]
t_masked = t[numpy.where(spt_scanmask == 1)[0]]
print "Done scan flag masking"

polynomial_coeff = numpy.transpose(numpy.polyfit(t_masked, numpy.transpose(spt_watts_masked), 2))
spt_watts_masked_corrected = spt_watts_masked - (polynomial_coeff[:, 2, numpy.newaxis] * numpy.ones(t_masked.shape[0])) - (polynomial_coeff[:, 1, numpy.newaxis] * t_masked) - (polynomial_coeff[:, 0, numpy.newaxis] * t_masked * t_masked)
spt_watts_uncorrected = spt_watts
spt_watts = spt_watts - (polynomial_coeff[:, 2, numpy.newaxis] * numpy.ones(t.shape[0])) - (polynomial_coeff[:, 1, numpy.newaxis] * t) - (polynomial_coeff[:, 0, numpy.newaxis] * t * t)
print "Done polynomial subtraction"

pyplot.plot(t, spt_watts_uncorrected[0])
pyplot.plot(t, spt_watts_uncorrected[1])
pyplot.plot(t, spt_watts[0])
pyplot.plot(t, spt_watts[1])
pyplot.xlabel("Time [s]")
pyplot.ylabel("Power [W]")
pyplot.legend(("Ch 1", "Ch 2", "Ch 1 (after)", "Ch 2 (after)"), 'upper right', fancybox=True)
pyplot.savefig("polynomial.png")
pyplot.clf()

pyplot.plot(t, spt_watts[0])
pyplot.plot(t, spt_watts[1])
pyplot.xlabel("Time [s]")
pyplot.ylabel("Power [W]")
pyplot.legend(("Ch 1", "Ch 2"), 'upper right', fancybox=True)
pyplot.savefig("corr.png")
pyplot.clf()

pyplot.scatter(spt_watts[0,:], spt_watts[1,:])
pyplot.xlim(-3e-14, 3e-14)
pyplot.ylim(-3e-14, 3e-14)
pyplot.xlabel("Ch 1 Power [W]")
pyplot.ylabel("Ch 2 Power [W]")
pyplot.savefig("corr2.png")
pyplot.clf()

cross_spec = numpy.dot(spt_watts_masked_corrected, numpy.transpose(spt_watts_masked_corrected))
cross_spec = cross_spec / spt_watts_masked_corrected.shape[1]
eigenvalues, eigenvectors = numpy.linalg.eigh(cross_spec)
eigenvectors = eigenvectors[:, numpy.argsort(eigenvalues)[::-1]]
eigenvalues = eigenvalues[numpy.argsort(eigenvalues)[::-1]]
print "Eigenvalues calculated"

pyplot.yscale("log")
pyplot.bar(numpy.arange(eigenvalues.shape[0]), eigenvalues)
pyplot.xlabel("Eigenvalue #")
pyplot.ylabel("Eigenvalue")
pyplot.savefig("eigenvalues.png")
pyplot.clf()
print eigenvectors[:,0]

spt_watts_pca = spt_watts
for i in range(5):
  spt_watts_pca = spt_watts_pca - numpy.dot(eigenvectors[:,i], spt_watts) * eigenvectors[:,i,numpy.newaxis]
print "Done PCA"

pyplot.plot(t, spt_watts[0])
pyplot.plot(t, spt_watts_pca[1])
pyplot.xlabel("Time [s]")
pyplot.ylabel("Power [W]")
pyplot.legend(("Ch 1 (before)", "Ch 1 (after)"), 'upper right', fancybox=True)
pyplot.savefig("corr3.png")
pyplot.clf()

pyplot.scatter(spt_watts_pca[0,:], spt_watts_pca[1,:])
pyplot.xlim(-3e-14, 3e-14)
pyplot.ylim(-3e-14, 3e-14)
pyplot.xlabel("Ch 1 Power [W]")
pyplot.ylabel("Ch 2 Power [W]")
pyplot.savefig("corr4.png")
pyplot.clf()

fft_len = int(2**math.floor(math.log(splitpoints_minlength,2.0)))
fft_hanning = numpy.hanning(fft_len)
fft_freqs = numpy.arange(fft_len/2 + 1) / delta_t / fft_len
spt_watts_splitbytime = numpy.array([spt_watts_pca[:, i:i+fft_len] for i in splitpoints_start])
spt_watts_splitbytime_o = numpy.array([spt_watts[:, i:i+fft_len] for i in splitpoints_start])
print "Split by time"

spt_fft = numpy.fft.rfft(spt_watts_splitbytime[:, 0:fft_len] * fft_hanning)
spt_fft_o = numpy.fft.rfft(spt_watts_splitbytime_o[:, 0:fft_len] * fft_hanning)

spt_fft_chan0 = spt_fft[:,0,:] * numpy.conjugate(spt_fft[:,0,:])
spt_fft_chan0 = numpy.sum(spt_fft_chan0, 0)
spt_fft_chan0 = spt_fft_chan0 * 2 * delta_t / numpy.sum(fft_hanning**2) / splitpoints_start.size

spt_fft_chan0_o = spt_fft_o[:,0,:] * numpy.conjugate(spt_fft_o[:,0,:])
spt_fft_chan0_o = numpy.sum(spt_fft_chan0_o, 0)
spt_fft_chan0_o = spt_fft_chan0_o * 2 * delta_t / numpy.sum(fft_hanning**2) / splitpoints_start.size

pyplot.plot(fft_freqs, numpy.sqrt(numpy.abs(spt_fft_chan0)))
pyplot.plot(fft_freqs, numpy.sqrt(numpy.abs(spt_fft_chan0_o)))
pyplot.legend(("Ch 1 (corrected)", "Ch 1 (original)"), 'upper right', fancybox=True)
pyplot.xlabel("Frequency [Hz]")
pyplot.ylabel("Power spectral density (ch1) [W/sqrt(Hz)]")
pyplot.xscale("log")
pyplot.yscale("log")
pyplot.savefig("psd.png")










