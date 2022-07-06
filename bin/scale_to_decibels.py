#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Convert an image to decibels.

FUTURE IMPROVEMENTS

TESTING STATUS
Tested.
'''

### IMPORT MODULES ---
import argparse
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from ImageIO import load_gdal_dataset, save_gdal_dataset
from ImageMasking import create_mask
from GeoFormatting import DS_to_extent
from ColorBalancing import scale_to_power_quantity_decibels
from ImageViewing import raster_multiplot



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
        help='Reference power. ([0], 1, 2, ..., minimum, mean, median, maximum).')
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



### REFERENCE POWER ---
def determine_reference_power(P0inpt, img, mask, verbose=False):
    '''
    Determine the reference power P0 directly or indirectly from the user
     input. Use the image values if necessary.
    '''
    if type(P0inpt) == int:
        # User-specified value
        P0 = P0inpt
        if inps.verbose == True:
            print('Interpreting P0 as integer')

    elif P0inpt == 'minimum':
        # Use smallest image value
        P0 = np.ma.array(img, mask=(mask==0)).min()

    elif P0inpt == 'mean':
        # Use mean image value
        P0 = np.ma.array(img, mask=(mask==0)).mean()

    elif P0inpt == 'median':
        # Use mean image value
        P0 = np.median(np.ma.array(img, mask=(mask==0)))

    elif P0inpt == 'maximum':
        # Use mean image value
        P0 = np.ma.array(img, mask=(mask==0)).max()

    else:
        print('Specified P0 could not be determined: {}'.format(P0inpt))
        exit()


    # Report P0
    if inps.verbose == True:
        print('P0: {:f}'.format(P0))

    return P0



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

    # Determine reference power
    P0 = determine_reference_power(inps.P0, img, mask, verbose=inps.verbose)


    ## Scaling
    dBimg = scale_to_power_quantity_decibels(img, P0)


    ## Outputs
    # Save if requested
    if inps.outname:
        # Format output name
        confirm_outdir(outname)
        outname = confirm_outname_ext(outname, ['tif'])

        # Save to GDAL data set
        save_gdal_dataset(outname, dBimg, mask=mask, exDS=DS,
            verbose=inps.verbose)

    # Plot if requested
    if inps.plot == True:
        raster_multiplot([img, dBimg], ncols=2, mask=mask,
            cmap=inps.cmap, cbarOrient='auto')

        plt.show()
