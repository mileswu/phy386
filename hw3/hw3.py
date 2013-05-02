import numpy
import math
import matplotlib.pyplot as pyplot
import pyfits

fits = pyfits.open("nras28.fits")
image = fits[0].data
t = numpy.arange(image.shape[0])
print "Loaded data"

def load(image, quadrant):
  if quadrant == 0:
    image_subset = image[:, :, 0:512]
  if quadrant == 1:
    image_subset = image[:, :, 1023:511:-1]

  image_subset_shape = image_subset.shape
  
  image_subset = numpy.reshape(image_subset, (image_subset_shape[0], image_subset_shape[1]*image_subset_shape[2]))
  polycoeff = numpy.polyfit(t, image_subset, 1)
  print "Done polyfit"
  
  image_repo = numpy.reshape(polycoeff[0,:], (image_subset_shape[1], image_subset_shape[2]) )
  
  pyplot.hist(numpy.reshape(image_repo, -1), 512, (-5, 5))
  pyplot.yscale("log")
  pyplot.xlabel("Slope")
  pyplot.savefig("hw3-q%d-hist.png" % quadrant)
  pyplot.clf()
  
  pyplot.imshow(image_repo[0:512,0:512], vmin=-1.,vmax=0.)
  pyplot.savefig("hw3-q%d-darkcurrent.png" % quadrant)
  pyplot.clf()
  
  image_repo = numpy.reshape(image_repo, -1)
  image_repo_filtered = numpy.copy(image_repo)
  hotcoldpixels = numpy.append(numpy.where(image_repo > 0)[0], numpy.where(image_repo < -1)[0])
  for i in hotcoldpixels:
    if i < (len(image_repo) - 10) and i >= 10:
      image_repo_filtered[i] = numpy.median(image_repo[i-10:i+10])
    else:
      print "Edge case, skipping"
  print "Done Filtering"
  
  pyplot.imshow(numpy.reshape(image_repo_filtered, (image_subset_shape[1], image_subset_shape[2]))[0:512,0:512], vmin=-1., vmax=0.)
  pyplot.savefig("hw3-q%d-darkcurrent-filtered.png" % quadrant)
  pyplot.clf()

  image_repo = numpy.reshape(image_repo, (image_subset_shape[1], image_subset_shape[2]))
  image_repo_filtered = numpy.reshape(image_repo_filtered, (image_subset_shape[1], image_subset_shape[2]))

  return image_repo,image_repo_filtered

image_repo0, image_repo0_filtered = load(image, 0)
image_repo1, image_repo1_filtered  = load(image, 1)
shape = image_repo0.shape

datawithblanks = numpy.insert(image_repo0_filtered, numpy.repeat(512, 12), numpy.zeros(12), axis=1)
fft_len = len(numpy.reshape(datawithblanks, -1))
fft_hanning = numpy.hanning(fft_len)
fft_freqs = numpy.arange(fft_len/2 + 1)

image_fft0_filtered = numpy.fft.rfft(numpy.reshape(datawithblanks, -1) * fft_hanning)
ps0_filtered = image_fft0_filtered * numpy.conjugate(image_fft0_filtered) * 2.0 / numpy.sum(fft_hanning**2)
pyplot.plot(fft_freqs, numpy.sqrt(numpy.abs(ps0_filtered)))
pyplot.yscale("log")
pyplot.ylabel("Frequency [px^-1]")
pyplot.xlabel("Linear power spectrum [arb]")
pyplot.xscale("log")
pyplot.savefig("hw3-q0-ps.png")
pyplot.clf()
print "Done FFT q0"

datawithblanks = numpy.insert(image_repo1_filtered, numpy.repeat(512, 12), numpy.zeros(12), axis=1)
image_fft1_filtered = numpy.fft.rfft(numpy.reshape(datawithblanks, -1) * fft_hanning)
cross_ps = image_fft0_filtered * numpy.conjugate(image_fft1_filtered) * 2.0 / numpy.sum(fft_hanning**2)
pyplot.plot(fft_freqs, numpy.sqrt(numpy.abs(cross_ps)))
pyplot.yscale("log")
pyplot.xscale("log")
pyplot.ylabel("Frequency [px^-1]")
pyplot.xlabel("Linear cross spectrum [arb]")
pyplot.savefig("hw3-q0q1-ps.png")
pyplot.clf()
print "Done FFT q0/q1"

combineddata = numpy.array([numpy.reshape(image_repo0_filtered,-1), numpy.reshape(image_repo1_filtered,-1)])
covar = numpy.dot(combineddata, numpy.transpose(combineddata))
covar = covar / combineddata.shape[1]
eigenvalues, eigenvectors = numpy.linalg.eigh(covar)
eigenvectors = eigenvectors[:, numpy.argsort(eigenvalues)[::-1]]
eigenvalues = eigenvalues[numpy.argsort(eigenvalues)[::-1]]
print "Eigenvalues and first eigenvector:"
print eigenvalues
print eigenvectors[:,0]

combined_pca = combineddata - numpy.dot(eigenvectors[:, 0], combineddata) * eigenvectors[:,0,numpy.newaxis]
image_repo0_filtered_subtracted = numpy.reshape(combined_pca[0], shape)
pyplot.imshow(image_repo0_filtered_subtracted[0:512,0:512], vmin=-0.2, vmax=0.2)
pyplot.savefig("hw3-q0-darkcurrent-filtered-subtracted.png")
pyplot.clf()
print "Done data PCA subtraction"

datawithblanks = numpy.insert(image_repo0_filtered_subtracted, numpy.repeat(512, 12), numpy.zeros(12), axis=1)
image_fft0_filtered_subtracted = numpy.fft.rfft(numpy.reshape(datawithblanks,-1) * fft_hanning)
ps0_filtered_subtracted = image_fft0_filtered_subtracted * numpy.conjugate(image_fft0_filtered_subtracted) * 2.0 / numpy.sum(fft_hanning**2)
pyplot.plot(fft_freqs, numpy.sqrt(numpy.abs(ps0_filtered)))
pyplot.plot(fft_freqs, numpy.sqrt(numpy.abs(ps0_filtered_subtracted)))
pyplot.yscale("log")
pyplot.xscale("log")
pyplot.ylabel("Frequency [px^-1]")
pyplot.xlabel("Linear power spectrum [arb]")
pyplot.legend(("Before subtraction", "After subtraction"), 'upper right', fancybox=True)
pyplot.savefig("hw3-q0-ps-filtered-subtracted.png")
pyplot.clf()
print "Done FFT q0 subtracted"
