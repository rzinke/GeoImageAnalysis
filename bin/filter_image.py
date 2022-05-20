#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Apply a specified filter to a GDAL compatible image.

FUTURE IMPROVEMENTS
    Frequency filter capabilities

TESTING STATUS
Tested.
'''

### IMPORT MODULES ---
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ImageIO import confirm_outdir, confirm_outname_ext, load_gdal_dataset, save_gdal_dataset
from ImageMasking import create_mask
from GeoFormatting import parse_transform, DS_to_extent
from ImageViewing import plot_raster


### PARSER ---
Description = '''Apply a specified filter to a GDAL compatible image.'''

Examples = ''''''

def createParser():
    parser = argparse.ArgumentParser(description=Description,
        formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    InputArgs = parser.add_argument_group('INPUTS')
    InputArgs.add_argument(dest='imgFile', type=str,
        help='Image file name.')
    InputArgs.add_argument('-b','--band', dest='bandNb', type=int, default=1,
        help='Image band number to display.')
    InputArgs.add_argument('-m','--mask', dest='maskArgs', nargs='+', type=str, default=None,
        help='Arguments for masking values/maps. ([None]).')
    InputArgs.add_argument('-f','--filter-type', dest='filtType', type=str, required=True,
        choices=['mean', 'blur', 'gauss', 'edge', 'sharpen', 'grad'],
        help='Filter type.')
    InputArgs.add_argument('-w','--kernel-width', dest='kernelWidth', type=int, required=True,
        help='Kernel width (pixels).')
    InputArgs.add_argument('--sigma', dest='kernelSigma', type=float, default=1.0,
        help='Standard deviation of Gaussian kernel ([1]).')
    InputArgs.add_argument('--intensity', dest='sharpenIntensity', type=float, default=1.0,
        help='Intensity of sharpening kernel ([1]).')

    OutputArgs = parser.add_argument_group('OUTPUTS')
    OutputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true',
        help='Verbose mode.')
    OutputArgs.add_argument('-p','--plot', dest='plot', action='store_true',
        help='Plot results.')
    OutputArgs.add_argument('-o','--outname', dest='outname', type=str, default=None,
        help='Filtered map name.')

    return parser

def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)



### FILTERING ---
def format_image(img, mask, verbose=False):
    '''
    Format the image before filtering to replace NaNs and masked values with
     zeros.
    '''
    if verbose == True: print('Pre-formatting image')

    # Replace NaNs with zeros
    img[np.isnan(img) == 1] = 0

    # Replace mask values
    img[mask==0] = 0

    return img


def filter_image(img, mask, dx, dy,
            filtType, kernelWidth, kernelSigma, sharpenIntensity,
            verbose=False):
    ''' Apply the specified filter to an image. '''

    if filtType == 'mean':
        from LinearFiltering import mean_filter
        filtImg = mean_filter(img, kernelWidth)

    elif filtType == 'blur':
        from LinearFiltering import blur
        filtImg = blur(img)

    elif filtType == 'gauss':
        from LinearFiltering import gauss_filter
        filtImg = gauss_filter(img, kernelWidth, kernelSigma)

    elif filtType == 'edge':
        from LinearFiltering import enhance_edges
        filtImg = enhance_edges(img)

    elif filtType == 'sharpen':
        from LinearFiltering import sharpen
        filtImg = sharpen(img)

    elif filtType == 'grad':
        from LinearFiltering import ImageGradient
        gradient = ImageGradient(img)

    return filtImg



### PLOTTING ---
def plot_images(img, fimg, mask, extent, verbose=False):
    ''' Plot the input and output images. '''
    if verbose == True: print('Plotting inputs and results')

    # Parameters
    cbarOrient = 'auto'

    # Plot input image
    fig0, ax0 = plot_raster(img, mask=mask, extent=extent,
        cmap='viridis', cbarOrient=cbarOrient,
        minPct=1, maxPct=99)
    ax0.set_title('Original image')

    # Plot filtered image
    fig0, ax0 = plot_raster(fimg, mask=mask, extent=extent,
        cmap='viridis', cbarOrient=cbarOrient,
        minPct=1, maxPct=99)
    ax0.set_title('Filtered image')    



### MAIN ---
if __name__ == '__main__':
    ## Inputs
    # Gather arguments
    inps = cmdParser()


    ## Load data
    # Load image data set
    DS = load_gdal_dataset(inps.imgFile, verbose=inps.verbose)
    _, dx, _, _, _, dy = DS.GetGeoTransform()
    extent = DS_to_extent(DS, verbose=inps.verbose)
    img = DS.GetRasterBand(inps.bandNb).ReadAsArray()

    # Create mask
    mask = create_mask(img, inps.maskArgs)

    # Pre-format image
    img = format_image(img, mask, verbose=inps.verbose)


    ## Filtering
    filtImg = filter_image(img, mask,
        dx = dx, dy = -dy,
        filtType=inps.filtType,
        kernelWidth=inps.kernelWidth,
        kernelSigma=inps.kernelSigma,
        sharpenIntensity=inps.sharpenIntensity,
        verbose=inps.verbose)


    ## Outputs
    # Save to file
    if inps.outname is not None:
        confirm_outdir(inps.outname)
        outname = confirm_outname_ext(inps.outname, ext=['tif'])
        save_gdal_dataset(outname, filtImg, mask=mask, exDS=DS, verbose=inps.verbose)

    # Plot if requested
    if inps.plot == True:
        plot_images(img, filtImg, mask, extent, verbose=inps.verbose)


    plt.show()