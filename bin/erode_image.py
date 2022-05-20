#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Erode an image based on mask values.

FUTURE IMPROVEMENTS


TESTING STATUS
Tested.
'''

### IMPORT MODULES ---
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ImageIO import confirm_outdir, confirm_outname_ext, load_gdal_dataset, save_gdal_dataset
from ImageMasking import create_mask
from GeoFormatting import DS_to_extent
from ErosionDilation import erode_image
from ImageViewing import plot_raster


### PARSER ---
Description = '''Erode an image based on mask values.'''

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
    InputArgs.add_argument('-w','--window-size', dest='windowSize', type=int, default=3,
        help='Width of window used in erosion filter. ([3], 5, 6, ...)')
    InputArgs.add_argument('-n','--max-nans', dest='maxNaNs', type=int, default=1,
        help='Max number of NaNs allowed per patch. (0, [1], 2, 3, ...)')
    InputArgs.add_argument('-i','--iterations', dest='iterations', type=int, default=1,
        help='Number of iterations. ([1], 2, ...)')

    OutputArgs = parser.add_argument_group('OUTPUTS')
    OutputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true',
        help='Verbose mode.')
    OutputArgs.add_argument('-p','--plot', dest='plot', action='store_true',
        help='Plot results.')
    OutputArgs.add_argument('-o','--outname', dest='outName', type=str, default='Eroded',
        help='Eroded map name.')

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
    extent = DS_to_extent(DS, verbose=inps.verbose)
    img = DS.GetRasterBand(inps.bandNb).ReadAsArray()

    # Create mask
    assert inps.maskArgs is not None, 'Mask argumets must be provided'
    mask = create_mask(img, inps.maskArgs)

    # Pre-format image
    img = format_image(img, mask, verbose=inps.verbose)


    ## Filtering
    erodedImg, erosionMask = erode_image(img, mask,
        windowSize=inps.windowSize, maxNaNs=inps.maxNaNs, iterations=inps.iterations,
        verbose=inps.verbose)


    ## Outputs
    # Save to file
    outName = confirm_outname_ext(inps.outName, ext=['tif'])
    confirm_outdir(inps.outName)
    save_gdal_dataset(outName, erodedImg, mask=erosionMask, exDS=DS,
        verbose=inps.verbose)

    # Plot if requested
    if inps.plot == True:
        fig0, ax0 = plot_raster(img, mask=mask, extent=extent,
            minPct=1, maxPct=99, cbarOrient='auto')
        ax0.set_title('Original image')

        figE, axE = plot_raster(erodedImg, mask=erosionMask, extent=extent,
            minPct=1, maxPct=99, cbarOrient='auto')
        axE.set_title('Eroded image')


    plt.show()