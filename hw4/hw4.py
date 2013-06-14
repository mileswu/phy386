import numpy
import math
import matplotlib.pyplot as pyplot
import pyfits

fits = pyfits.open("nr29.fits")
image = fits[0].data
t = numpy.arange(image.shape[0])
print "Loaded data"

def load(image, quadrant):
  if quadrant == 1:
    image_subset = image[:, :, 1024:1536]
  if quadrant == 2:
    image_subset = image[:, :, 1535:1023:-1]
  if quadrant == 3:
    image_subset = image[:, :, 1536:2048]

  image_subset_shape = image_subset.shape
  
  image_subset = numpy.reshape(image_subset, (image_subset_shape[0], image_subset_shape[1]*image_subset_shape[2]))
  polycoeff = numpy.polyfit(t, image_subset, 1)
  print "Done polyfit"

  polycoeff = polycoeff[0,:]
  polycoeff = polycoeff - numpy.median(polycoeff)
  image_repo = numpy.reshape(polycoeff, (image_subset_shape[1], image_subset_shape[2]) )
  
  pyplot.hist(numpy.reshape(image_repo, -1), 512, (-5, 5))
  pyplot.yscale("log")
  pyplot.xlabel("Slope")
  pyplot.savefig("hw4-q%d-hist.png" % quadrant)
  pyplot.clf()
  
  pyplot.imshow(image_repo[0:512,0:512], vmin=-0.5,vmax=0.5)
  pyplot.savefig("hw4-q%d-darkcurrent.png" % quadrant)
  pyplot.clf()
  
  image_repo = numpy.reshape(image_repo, -1)
  image_repo_filtered = numpy.copy(image_repo)
  hotcoldpixels = numpy.append(numpy.where(image_repo > 1.0)[0], numpy.where(image_repo < -1.0)[0])
  for i in hotcoldpixels:
    if i < (len(image_repo) - 10) and i >= 10:
      image_repo_filtered[i] = numpy.median(image_repo[i-10:i+10])
  print "Done Filtering"
  
  pyplot.imshow(numpy.reshape(image_repo_filtered, (image_subset_shape[1], image_subset_shape[2]))[0:512,0:512], vmin=-0.5, vmax=0.5)
  pyplot.savefig("hw4-q%d-darkcurrent-filtered.png" % quadrant)
  pyplot.clf()

  image_repo = numpy.reshape(image_repo, (image_subset_shape[1], image_subset_shape[2]))
  image_repo_filtered = numpy.reshape(image_repo_filtered, (image_subset_shape[1], image_subset_shape[2]))

  return image_repo,image_repo_filtered

image2 = load(image, 2)[1]
image3 = load(image, 3)[1]
shape = image2.shape
image = image3 - image2

row_average = numpy.sum(image, 1) / shape[1]
image = image - row_average[:, numpy.newaxis]
print "Done subtraction"

pyplot.imshow(image, vmin=-0.2, vmax=1.0)
pyplot.savefig("hw4-image.png")
pyplot.clf()

lumi = numpy.sum(image, 0) / shape[0]
pyplot.plot(lumi)
pyplot.xlabel("Column [px]")
pyplot.ylabel("Luminosity [arb]")
pyplot.savefig("hw4-lumi.png")
pyplot.clf()

jump_delta = numpy.sum(numpy.abs(numpy.diff(lumi)))/2.0/(lumi.shape[0]-1)
correction = numpy.reshape(numpy.repeat([[jump_delta, -jump_delta]], lumi.shape[0]/2, 0), -1)
print "Done jump correction"

image = image + correction
lumi_jumpcorrected = numpy.sum(image, 0) / shape[0]
pyplot.plot(lumi_jumpcorrected)
pyplot.xlabel("Column [px]")
pyplot.ylabel("Luminosity [arb]")
pyplot.savefig("hw4-lumi-corrected.png")
pyplot.clf()

fft_win = numpy.hanning(shape[1])
fft = numpy.fft.rfft(image * fft_win)
fft_freqs = numpy.arange(fft.shape[1])
mag2 = numpy.sum(numpy.abs(fft)**2, 0) #mag^2 of each column
arg = numpy.angle(numpy.sum(fft / numpy.roll(fft, 1, 0), 0)) #divide row i by row i+1 and sum each column
coeff = numpy.polynomial.polynomial.polyfit(fft_freqs[0:130], arg[0:130], 1, w=mag2[0:130])

pyplot.plot(arg)
pyplot.plot(fft_freqs*coeff[1] + coeff[0])
pyplot.xlabel("Row frequency space [px^-1]")
pyplot.ylabel("Arg. [rad]")
pyplot.legend(("Data", "Fit"), 'upper right', fancybox=True)
pyplot.savefig("hw4-rotation-fit.png")
pyplot.clf()
print "Done argument fit for rotation"
print "Rotation factor: %f" % coeff[1]

fft = numpy.fft.rfft(image)
rotation_factor_row = numpy.arange(fft.shape[0]) - fft.shape[0]/2
rotation_factor_column = numpy.arange(fft.shape[1])
rotation_factor = numpy.exp(-coeff[1]*numpy.outer(rotation_factor_row, rotation_factor_column)*1.0j)
image_rotated = numpy.fft.irfft(fft*rotation_factor)
print "Done image rotation"

pyplot.figure(figsize=(10,5))
pyplot.subplot(121)
pyplot.imshow(image, vmin=-0.2, vmax=1.0, aspect='auto')
pyplot.subplot(122)
pyplot.imshow(image_rotated, vmin=-0.2, vmax=1.0, aspect='auto')
pyplot.savefig("hw4-image-rotation.png")
pyplot.clf()

lumi_rotated = numpy.sum(image_rotated, 0) / shape[0]
pyplot.plot(lumi_jumpcorrected)
pyplot.plot(lumi_rotated)
pyplot.xlabel("Column [px]")
pyplot.ylabel("Luminosity [arb]")
pyplot.legend(("Before rotation", "After rotation"), 'upper right', fancybox=True)
pyplot.savefig("hw4-lumi-rotated.png")
pyplot.clf()

#Chosen galaxy in column 257
weighting = numpy.kaiser(6,3)
spectrum = numpy.sum(image[:, 254:260] * weighting, 1)[::-1]

weighting = numpy.kaiser(40,8)
weighting = weighting / numpy.sum(weighting)

freq = numpy.arange(spectrum.shape[0])
freq_smoothed = numpy.arange(spectrum.shape[0])[19:-20]
spectrum_smoothed = numpy.convolve(spectrum, weighting, 'valid')
pyplot.plot(freq_smoothed, spectrum_smoothed)
pyplot.xlabel("Frequency for Galaxy in Column 257 [px]")
pyplot.ylabel("Luminosity [arb]")
pyplot.savefig("hw4-spectrum.png")
pyplot.clf()
print "Obtained spectrum from Galaxy col 257"

templates = numpy.transpose(numpy.load("galaxy_templates.npz")['gal_spectra'])
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
fitting_funcs = []
params = []
residuals = []
for i in range(1,5):
  template_f = numpy.log(templates[0])[::-1]
  template_flux = templates[i][::-1]
  template_function = interp1d(template_f, template_flux, bounds_error=False, fill_value=0)
  fittingfunc = lambda v, a, z, c1, c2, template_function=template_function: a*template_function(v/(1.0+z) + c1) + c2
  param, _ = curve_fit(fittingfunc, freq, spectrum, p0=(0.2, 800, 31.64, 0.0))
  fitting_funcs.append(fittingfunc)
  params.append(param)
  residuals.append(numpy.sum((fittingfunc(freq, *param) - spectrum)**2))

print "Done curve fitting"
print "Residuals are:"
print residuals

pyplot.plot(freq_smoothed, spectrum_smoothed)
for i in range(4):
  pyplot.plot(freq, fitting_funcs[i](freq, *params[i]))
pyplot.xlabel("Frequency for Galaxy in Column 257 [px]")
pyplot.ylabel("Luminosity after curve fitting [arb]")
pyplot.legend(("Data", "1068", "m82", "orp220", "ngc6946"), 'upper right', fancybox=True)
pyplot.savefig("hw4-spectrum-fit.png")
pyplot.clf()
  
pyplot.plot(freq_smoothed, spectrum_smoothed)
for i in range(4):
  pyplot.plot(freq, fitting_funcs[i](freq, 0.2, 800, 31.64, 0))
pyplot.xlabel("Frequency for Galaxy in Column 257 [px]")
pyplot.ylabel("Luminosity before curve fitting [arb]")
pyplot.legend(("Data", "1068", "m82", "orp220", "ngc6946"), 'upper right', fancybox=True)
pyplot.savefig("hw4-spectrum-unfit.png")
pyplot.clf()
