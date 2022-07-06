'''
Functions for single band color balancing.
'''

### IMPORT MODULES ---
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate



### IMAGE STATISTICS ---
class ImageStats:
    def __init__(self, img):
        '''
        Compute the basic statistics associated with an image.
        '''
        # Flatten image values
        img = img.flatten()

        # Basic statistics
        self.min = img.min()
        self.max = img.max()
        self.mean = img.mean()
        self.median = np.median(img)

    def report(self):
        '''
        Report statistics.
        '''
        print('Extrema: {:.3f}, {:.3f}'.format(self.min, self.max))
        print('Mean: {:.3f}'.format(self.mean))
        print('Median: {:.3f}'.format(self.median))



### HISTOGRAM ---
class ImageHistogram:
    def __init__(self, img, nbins=256):
        '''
        Create a histogram of the image.
        '''
        # Flatten image values
        img = img.flatten()

        # Create histogram
        self.hvals, hedges = np.histogram(img, bins=nbins)

        # Bin centeres
        self.hbins = (hedges[:-1] + hedges[1:])/2

        # Build interpolation function
        self.intrp = interpolate.interp1d(self.hbins, self.hvals, kind='linear')

    def show(self):
        '''
        Plot histogram.
        '''
        # Spawn figure and axis
        fig, ax = plt.subplots()

        # Plot histogram
        markerline, stemlines, baseline = plt.stem(self.hbins, self.hvals,
            linefmt='r', markerfmt='', use_line_collection=True)
        stemlines.set_linewidths(None)
        baseline.set_linewidth(0)
        ax.plot(self.hbins, self.hvals, 'k', linewidth=2)



### COLOR BALANCING ---
def norm_colors(img):
    '''
    Stretch colors to values between 0 and 255.
    '''
    img = img - img.min()
    img = 255 * img/img.max()

    return img


def logistic_transform(img):
    '''
    Apply a logistic function such that negative values are close to zero and
     positive values are close to one.
    '''
    # Logistic
    img = 1/(1+np.exp(-img))


def equalize_image(img):
    ''' Equalize the color range of the image. '''
    # Compute histogram
    hist = ImageHistogram(img)

    # Build transform
    x = np.arange(0, 256)
    y = np.cumsum(hist.hvals)  # integrate
    y[0] = 0  # set min value to 0
    y = 255*y/y.max()  # set max value to 255
    intp = interpolate.interp1d(x, y, kind='linear')
    img = intp(img)

    return img


def scale_to_power_quantity_decibels(img, P0):
    '''
    Scale the pixel intensities to decibels based on the power quanitities,
     according to the formula:
      Lp = 10.log10(P/P0) dB
    '''
    return 10*np.log10(img/P0)
