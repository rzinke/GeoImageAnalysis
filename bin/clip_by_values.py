#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Clip image based on values.

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
from ImageViewing import image_stats, image_percentiles, plot_raster


### PARSER ---
Description = '''Mask a raster by values.'''

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
    InputArgs.add_argument('-minPct','--min-percent', dest='minPct', type=float, default=0,
        help='Minimum percent clip value ([None]).')
    InputArgs.add_argument('-maxPct','--max-percent', dest='maxPct', type=float, default=100,
        help='Maximum percent clip value ([None]).')
    InputArgs.add_argument('-vmin','--min-value', dest='vmin', type=float, default=None,
        help='Minimum clip value ([None]).')
    InputArgs.add_argument('-vmax','--max-value', dest='vmax', type=float, default=None,
        help='Maximum clip value ([None]).')
    InputArgs.add_argument('--replace-values', dest='replaceValues', action='store_true',
        help='Replace clipped pixels with respective clip values, rather than masking them.')

    HistogramArgs = parser.add_argument_group('HISTOGRAM')
    HistogramArgs.add_argument('-hist','--histogram', dest='plotHist', action='store_true')

    OutputArgs = parser.add_argument_group('OUTPUTS')
    OutputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true',
        help='Verbose mode.')
    OutputArgs.add_argument('-p','--plot', dest='plot', action='store_true',
        help='Plot results.')
    OutputArgs.add_argument('-o','--outname', dest='outname', type=str, default='Clipped',
        help='Clipped map name.')

    return parser

def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)



### AUXILIARY FUNCTIONS ---
def clip_values(img, mask, vmin, vmax, minPct, maxPct, replaceValues,
    verbose=False):
    ''' Clip or replace image values. '''
    # Copy mask for clipping
    clippingMask = mask.copy()

    # Minimum value
    if inps.vmin is not None:
        vmin = inps.vmin
    else:
        vmin, _ = image_percentiles(img, inps.minPct, 100)

    # Nullify or replace minimum values
    if replaceValues == False:
        clippingMask[img < vmin] = 0
    elif replaceValues == True:
        img[img < vmin] = vmin

    # Maximum value
    if inps.vmax is not None:
        vmax = inps.vmax
    else:
        _, vmax = image_percentiles(img, 0, inps.maxPct)

    # Nullify or replace maximum values
    if replaceValues == False:
        clippingMask[img > vmax] = 0
    elif replaceValues == True:
        img[img > vmax] = vmax

    # Report if requested
    if inps.verbose == True:
        print('Clipping image to range {:.4f}, {:.4f}'.format(vmin, vmax))

    return img, clippingMask



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
    img[np.isnan(img) == 1] = 0  # replace NaNs

    # Create mask
    mask = create_mask(img, inps.maskArgs)

    # Report original values if requested
    image_stats(img, mask, verbose=inps.verbose)


    ## Clip values
    img, clippingMask = clip_values(img, mask=mask,
        vmin=inps.vmin, vmax=inps.vmax,
        minPct=inps.minPct, maxPct=inps.maxPct,
        replaceValues=inps.replaceValues,
        verbose=inps.verbose)


    ## Outputs
    # Save to file
    outname = confirm_outname_ext(inps.outname, ext=['tif'])
    confirm_outdir(inps.outname)
    save_gdal_dataset(outname, img, mask=clippingMask, exDS=DS,
        verbose=inps.verbose)

    # Plot if requested
    if inps.plot == True:
        fig0, ax0 = plot_raster(img, mask=mask, extent=extent, cbarOrient='auto')
        ax0.set_title('Original image')

        figClip, axClip = plot_raster(img, mask=clippingMask, extent=extent,
            cbarOrient='auto')
        axClip.set_title('Clipped image')


    plt.show()