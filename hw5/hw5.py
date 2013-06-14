import numpy
import pyfits
import matplotlib.pyplot as pyplot
import scipy.ndimage as ndimage

fits = pyfits.open("final_coadd_ra5h30dec-55_masked_proj0_150.fits")
signal = fits[3].data[0][0]
noise = fits[4].data[0][0]
weight = fits[6].data[0][0]

shape = signal.shape
weight = numpy.roll(weight, shape[0]/2, 0)
weight = numpy.roll(weight, shape[1]/2, 1)

n = 2048
signal = signal[shape[0]/2-n/2:shape[0]/2+n/2, shape[1]/2-n/2:shape[1]/2+n/2]
noise = noise[shape[0]/2-n/2:shape[0]/2+n/2, shape[1]/2-n/2:shape[1]/2+n/2]
weight = ndimage.zoom(weight, (float(n)/float(shape[0]), float(n)/float(shape[1])), order = 2)
shape = signal.shape
print "Data cropped/resized"

pyplot.hist(numpy.reshape(signal, -1), 512, (-0.001, 0.001))
pyplot.xlabel("Values")
pyplot.ylabel("Entries")
pyplot.yscale("log")
pyplot.savefig("hw5-hist.png")
pyplot.clf()

pyplot.imshow(signal, vmin=-0.00015, vmax=0.00015)
pyplot.savefig("hw5-signal.png")
pyplot.clf()
pyplot.imshow(signal, vmin=0.00025, vmax=0.001)
pyplot.savefig("hw5-signal2.png")
pyplot.clf()
pyplot.imshow(noise, vmin=-0.00015, vmax=0.00015)
pyplot.savefig("hw5-noise.png")
pyplot.clf()
pyplot.imshow(weight)
pyplot.savefig("hw5-weight.png")
pyplot.clf()

signal = numpy.reshape(signal, -1)
x_range = numpy.arange(0.0, 2048.0)
y_range = numpy.arange(0.0, 2048.0)
num_removed = 0
while numpy.amax(signal) > 0.0003:
  num_removed += 1
  index = signal.argmax()
  x_index = index/shape[0]
  y_index = index % shape[0]
  r2 = numpy.reshape((y_range - y_index)**2 + ((x_range - x_index)**2)[:, numpy.newaxis], -1)
  inside = numpy.where(r2 < 8*8)[0]
  outside = numpy.where((r2 < 9*8*8) & (r2 > 8*8))[0]
  signal[inside] = numpy.median(signal[outside])
signal = numpy.reshape(signal, shape)
print "%d sources removed" % num_removed

pyplot.hist(numpy.reshape(signal, -1), 512, (-0.001, 0.001))
pyplot.xlabel("Values")
pyplot.ylabel("Entries")
pyplot.yscale("log")
pyplot.savefig("hw5-hist2.png")
pyplot.clf()


hanning_win = numpy.hanning(shape[0])
hanning_win = numpy.outer(hanning_win, hanning_win)

signal_fft = numpy.fft.fft2(signal * hanning_win)
noise_fft = numpy.fft.fft2(noise * hanning_win)
signal_fft = numpy.abs(signal_fft)**2 - numpy.abs(noise_fft)**2
signal_fft = numpy.roll(signal_fft, shape[0]/2, 0)
signal_fft = numpy.roll(signal_fft, shape[1]/2, 1)

signal_fft = numpy.reshape(signal_fft, -1)
weight = numpy.reshape(weight, -1)
print "Done FFT"

max_r = 250
x_range = numpy.arange(-1024.0, 1024.0)
y_range = numpy.arange(-1024.0, 1024.0)
r2 = numpy.reshape(y_range**2 + (x_range**2)[:, numpy.newaxis], -1)
l = numpy.arange(max_r) * 360.0 * 60.0 * 2.0 / shape[0]

spec = numpy.zeros(max_r)
spec_weighted = numpy.zeros(max_r)
spec_weighted_n = numpy.zeros(max_r)
spec_n = numpy.zeros(max_r)
for i in range(shape[0]*shape[1]):
  index = int(numpy.sqrt(r2[i]))
  if(index >= max_r):
    continue
  spec_n[index] += 1
  spec[index] += signal_fft[i]

  if(weight[i] == 0):
    continue
  spec_weighted_n[index] += 1
  spec_weighted[index] += signal_fft[i] / weight[i]

for i in range(len(spec)):
  if(spec_n[i] != 0):
    spec[i] = spec[i] / spec_n[i]
  if(spec_weighted_n[i] != 0):
    spec_weighted[i] = spec_weighted[i] / spec_weighted_n[i]

spec = spec * shape[0] / 360.0 / 60.0 / 2.0
spec_weighted = spec_weighted * shape[0] / 360.0 / 60.0 / 2.0
print "Calcualted multipole spectrum"

pyplot.plot(l, spec_weighted)
pyplot.plot(l, spec)
pyplot.legend(("Weighted", "Unweighted"), 'upper right', fancybox=True)
pyplot.yscale("log")
pyplot.xscale("log")
pyplot.ylabel("Power [arb]")
pyplot.xlabel("Mulitpole")
pyplot.savefig("hw5-spec.png")
pyplot.clf()
