#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Convert an image to decibels.

FUTURE IMPROVEMENTS

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
from ImageViewing import scale_to_power_quantity_decibels



### PARSER ---
Description = '''
Convert the image values to decibels according to the formula:
  Lp = 10*log10(P/P0) dB
where P0 is some reference power
'''

Examples = ''''''


def createParser():
    parser = argparse.ArgumentParser(description=Description,
        formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    InputArgs = parser.add_argument_group('INPUTS')
    InputArgs.add_argument(dest='imgname', type=str,
        help='Image name.')
    InputArgs.add_argument('-p0','--reference-power', dest='P0', default=1,
        help='Reference power. ([0], 1, 2, ..., minimum, average).')
    InputArgs.add_argument('-m','--mask', dest='maskArgs', nargs='+', type=str, default=None,
        help='Arguments for masking values/maps. ([None]).')

    OutputArgs = parser.add_argument_group('OUTPUTS')
    OutputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true',
        help='Verbose mode.')
    OutputArgs.add_argument('-p','--plot', dest='plot', type=int, default=None,
        help='Plot components. (Provide number of components to plot).')
    OutputArgs.add_argument('-o','--outname', dest='outname', type=str, default=None,
        help='Principal components name.')

    return parser


def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)



### PCA ---
if __name__ == '__main__':
    ## Inputs
    # Gather arguments
    inps = cmdParser()


    ## Load data and format inputs
    # Load image
    DS = load_gdal_dataset(inps.imgname, verbose=inps.verbose)
    img = DS.GetRasterBand(1).ReadAsArray()  # measured power

    # Create mask
    mask = create_mask(img, inps.maskArgs, verbose=inps.verbose)

    # Determine reference power
    if type(inps.P0) == int:
        P0 = inps.P0


    ## Scaling
    Lp = scale_to_power_quantity_decibels(1000, P0)

    print(Lp)


    ## Outputs


    plt.show()
