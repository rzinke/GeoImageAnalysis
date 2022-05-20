#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Create mask from image.

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
from ImageViewing import plot_raster


### PARSER ---
Description = '''Create a mask from an image.'''

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

    OutputArgs = parser.add_argument_group('OUTPUTS')
    OutputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true',
        help='Verbose mode.')
    OutputArgs.add_argument('-p','--plot', dest='plot', action='store_true',
        help='Plot results.')
    OutputArgs.add_argument('-o','--outname', dest='outName', type=str, default='Mask',
        help='Mask name.')

    return parser

def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)



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
    mask = create_mask(img, inps.maskArgs, verbose=inps.verbose)


    ## Outputs
    # Save to file
    outName = confirm_outname_ext(inps.outName, ext=['tif'])
    confirm_outdir(inps.outName)
    save_gdal_dataset(outName, mask, exDS=DS, verbose=inps.verbose)

    # Plot if requested
    if inps.plot == True:
        fig0, ax0 = plot_raster(img, mask=mask, extent=extent, cbarOrient='auto')
        ax0.set_title('Original image')

        figMsk, axMsk = plot_raster(mask, extent=extent, cbarOrient='auto')
        axMsk.set_title('Mask')


    plt.show()
