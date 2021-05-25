# mnist

MNIST processing that takes the downloads from LeCun's site, resamples
them using scipy.ndimage.zoom to 32x32, and exports them into a
thresholded boolean dataset and a dataset where each pixel is a byte.
