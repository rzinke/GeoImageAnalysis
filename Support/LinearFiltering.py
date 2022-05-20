'''
SHORT DESCRIPTION
Create and apply linear filters to an image.

FUTURE IMPROVEMENTS

TESTING STATUS
Tested.
'''

### IMPORT MODULES ---
import numpy as np
from scipy import signal

from ColorBalancing import norm_colors



### FILTER APPLICATION ---
def apply_linear_filter(img, H, fft=False):
    '''
    Apply a linear filter to an image by convolution in the Fourier domain.
    INPUTS
        img is the image
        H is the filter kernel
    '''
    if fft == False:
        # Spatial domain convolution
        img = signal.convolve2d(img, H, 'same')

    elif fft == True:
        # Frequency domain convolution (faster)
        img = signal.fftconvolve(img, H, 'same')

    return img



### SPATIAL FILTER DESIGN ---
def mean_filter(img, w):
    '''
    Design a moving average filter kernel and apply.
    INPUTS
        w is the width of the kernel, in pixels
    '''
    # Boxcar kernel
    H = np.ones((w, w))/w**2

    # Apply filter to image
    img = apply_linear_filter(img, H)

    return img

def blur(img):
    ''' Apply a standard 3x3 blur. '''
    # Filter kernel
    H = np.array([[1/8, 1/4, 1/8],
                  [1/4,  1 , 1/4],
                  [1/8, 1/4, 1/8]])

    # Apply filter to image
    img = apply_linear_filter(img, H)

    return img

def gauss_filter(img, w, sigma):
    '''
    Design a Gaussian filter kernel and apply.
    INPUTS
        w is the width of the kernel, in pixels
        sigma is the standard deviation, in pixels
    '''
    # 1D Gaussian window
    h = signal.gaussian(w, sigma)

    # Outer product for 2D kernel
    H = np.dot(h.reshape(w,1), h.reshape(1,w))

    # Apply filter to image
    img = apply_linear_filter(img, H)

    return img

def enhance_edges(img):
    ''' Enhance edges in the image. '''
    # Filter kernel
    H = np.ones((3, 3))/-8
    H[1,1] = 1

    # Apply filter to image
    img = apply_linear_filter(img, H)

    return img

def sharpen(img, intensity):
    '''
    Sharpen an image by multiplying by a value and dividing by the mean of the
     surrounding pixels.
    '''
    # Filter kernel
    H = intensity * np.ones((3, 3))/-8
    H[1,1] = intensity + 1

    # Apply filter to image
    img = apply_linear_filter(img, H)

    return img

def shade(img, azimuth):
    '''
    Apply the shift and subtract method to produce an embossed look.
    Azimuth is given in compass degrees.
    '''
    # Convert azimuth to position
    azimuth = np.deg2rad(90 - azimuth)  # convert to unit circle radians
    j = np.round(np.cos(azimuth)) + 1  # x-position
    i = 1 - np.round(np.sin(azimuth))  # y-position

    # Filter kernel
    H = np.zeros((3, 3))
    H[1,1] = 1
    H[int(i), int(j)] = -1

    # Apply filter to image
    img = apply_linear_filter(img, H)

    return img



### METRICS ---
def standard_deviation(img):
    ''' Compute the standard deviation of a 3x3 group of pixels. '''
    # Filter kernel
    EX = mean_filter(img, 3)
    EX2 = mean_filter(img**2, 3)

    # Compute variance
    var = EX2 - EX**2
    var[var<0] = 0  # correct rounding error

    # Compute sandard deviation
    sd = np.sqrt(var)

    return sd



### SLOPES AND GRADIENTS ---
class ImageGradient:
    def __init__(self, img, filterType='sobel', dx=1, dy=1):
        '''
        Calculate a gradient across the image using the specified filter type.
        '''
        # Establish filter kernel
        H = self.__build_kernel__(filterType, dx, dy)

        # Compute gradient
        filtImg = apply_linear_filter(img, H)

        # Parse results
        self.dzdx = filtImg.real
        self.dzdy = filtImg.imag
        self.gradient = np.abs(filtImg)
        self.azimuth = np.angle(filtImg)

    def __build_kernel__(self, filterType, dx, dy):
        '''
        Build the filter based on the specified kernel type and pixel dimensions.
        '''
        # Check filter type
        filterType = filterType.lower()
        assert filterType in ['roberts', 'prewitt', 'sobel', 'scharr']

        # Build kernel
        if filterType == 'roberts':
            H = np.array([[ 0+1.j, 1+0.j],
                          [-1+0.j, 0-1.j]])
            H.real = H.real/(2*dx)
            H.imag = H.imag/(2*dy)
        elif filterType == 'prewitt':
            H = np.array([[1+1.j, 0+1.j, -1+1.j],
                          [1+0.j, 0+0.j, -1+0.j],
                          [1-1.j, 0-1.j, -1-1.j]])
            H.real = H.real/(6*dx)
            H.imag = H.imag/(6*dy)
        elif filterType == 'sobel':
            H = np.array([[1+1.j, 0+2.j, -1+1.j],
                          [2+0.j, 0+0.j, -2+0.j],
                          [1-1.j, 0-2.j, -1-1.j]])
            H.real = H.real/(8*dx)
            H.imag = H.imag/(8*dy)
        elif filterType == 'scharr':
            H = np.array([[ 3+3.j, 0+10.j, -3+3.j ],
                          [10+0.j,  0+0.j, -10+0.j],
                          [ 3-3.j, 0-10.j, -3-3.j ]])

        return H
