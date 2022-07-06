#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Transform values to 0-1 using logistic function.

FUTURE IMPROVEMENTS
 Pre-condition to avoid overflow

TESTING STATUS
In development.
'''

### IMPORT MODULES ---
import argparse
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from ImageIO import load_gdal_dataset, save_gdal_dataset
from ImageMasking import create_mask
from GeoFormatting import DS_to_extent
from ColorBalancing import logistic_transform
from ImageViewing import raster_multiplot



### PARSER ---
Description = '''
Transform image values to the range 0 - 1 using a logistic curve:
  1/(1+exp(-x))
'''

Examples = ''''''


def createParser():
    parser = argparse.ArgumentParser(description=Description,
        formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    InputArgs = parser.add_argument_group('INPUTS')
    InputArgs.add_argument(dest='imgname', type=str,
        help='Image name.')
    InputArgs.add_argument('-m','--mask', dest='maskArgs', nargs='+', type=str, default=None,
        help='Arguments for masking values/maps. ([None]).')

    OutputArgs = parser.add_argument_group('OUTPUTS')
    OutputArgs.add_argument('--cmap', dest='cmap', type=str, default='viridis',
        help='Colormap for plots. ([viridis]).')
    OutputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true',
        help='Verbose mode.')
    OutputArgs.add_argument('-p','--plot', dest='plot', action='store_true',
        help='Plot components. (Provide number of components to plot).')
    OutputArgs.add_argument('-o','--outname', dest='outname', type=str, default=None,
        help='Principal components name.')

    return parser


def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)



### MAIN ---
if __name__ == '__main__':
    ## Inputs
    # Gather arguments
    inps = cmdParser()


    ## Load data and format inputs
    # Load image
    DS = load_gdal_dataset(inps.imgname, verbose=inps.verbose)
    img = DS.GetRasterBand(1).ReadAsArray()  # measured power
    extent = DS_to_extent(DS)

    # Create mask
    mask = create_mask(img, inps.maskArgs, verbose=inps.verbose)


    ## Scaling
    squishedImg = logistic_transform(img)


    ## Outputs
    # Save if requested
    if inps.outname:
        # Format output name
        confirm_outdir(outname)
        outname = confirm_outname_ext(outname, ['tif'])

        # Save to GDAL data set
        save_gdal_dataset(outname, squishedImg, mask=mask, exDS=DS,
            verbose=inps.verbose)

    # Plot if requested
    if inps.plot == True:
        raster_multiplot([img, squishedImg], ncols=2, mask=mask,
            cmap=inps.cmap, cbarOrient='auto')

        plt.show()
