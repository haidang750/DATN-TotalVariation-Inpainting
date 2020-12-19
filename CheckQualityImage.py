import numpy as np
from skimage import measure

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	im1, im2 = _as_floats(imageA, imageB)
	return np.mean(np.square(im1 - im2), dtype=np.float64)

def _as_floats(im1, im2):
    float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
    im1 = np.asarray(im1, dtype=float_type)
    im2 = np.asarray(im2, dtype=float_type)
    return im1, im2

def SSIM(imageA, imageB):
    return measure.compare_ssim(imageA, imageB, multichannel=True)

def PSNR(imageA, imageB):
	err = mse(imageA, imageB)
	Imax = np.max(imageA)
	return 10 * np.log10((Imax ** 2) / err)

