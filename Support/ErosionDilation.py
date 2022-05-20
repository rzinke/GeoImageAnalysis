'''
SHORT DESCRIPTION
Erode or dilate an image based on pixel value thresholds.

FUTURE IMPROVEMENTS

TESTING STATUS
Need to test dilate_image
'''

### IMPORT MODULES ---
import numpy as np

from LinearFiltering import mean_filter



### EROSION/DILATION ---
def erode_image(img, mask, windowSize=3, maxNaNs=1, iterations=1, verbose=False):
    ''' Erode an image based on mask values. '''
    if verbose == True:
        print('Eroding image\n\twindow width: {:d}'.format(windowSize))

    # Setup
    erodedImg = img.copy()
    erosionMask = mask.copy()

    # Define erosion threshold, below which values will be masked
    erosionThreshold = 1-maxNaNs/windowSize**2
    if verbose == True: print('\tthreshold: {:.2f}'.format(erosionThreshold))

    # Erode image
    for i in range(iterations):
        if verbose == True: print('\tapplying erosion filter: {:d}'.format(i+1))

        # Dilate mask using mean filter
        erosionMask = mean_filter(erosionMask, windowSize)
        erosionMask[erosionMask < erosionThreshold] = 0  # binary mask

        # Erode image
        erodedImg = erodedImg * erosionMask

    return erodedImg, erosionMask


def dilate_image(img, mask, windowSize=3, maxNaNs=1, iterations=1, verbose=False):
    ''' Dilate an image based on mask values. '''
    if verbose == True:
        print('Dilating image\n\twindow width: {:d}'.format(windowSize))

    # Setup
    dilationValues = img.copy()
    dilatedImg = img.copy()
    dilatedMask = mask.copy()

    # Define dilation threshold, above which pixels will be dilated
    dilationThreshold = 1 - maxNaNs/windowSize**2

    # Dilate image
    for i in range(iterations):
        if verbose == True: print('\tapplying dilation filter: {:d}'.format(i+1))

        # Dilate mask using mean filter
        dilationMask = mean_filter(dilationMask, windowSize)
        dilationMask[dilationMask > dilationThreshold] = 1

        # Dilation values
        meanImg = mean_filter(dilationMask, windowSize)

        # Dilate images
        dilatedImg[(dilationMask > dilationThreshold) & (mask == 0)] = \
            dilationValues[(dilationMask > dilationThreshold) & (mask == 0)]

    return dilatedImg, dilatedMask
