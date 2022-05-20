#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Compute the semivariogram and semicovariogram for an image.

FUTURE IMPROVEMENTS


TESTING STATUS
Tested.
'''

### IMPORT MODULES ---
import argparse
import numpy as np
import matplotlib.pyplot as plt

from IOsupport import confirm_outdir, confirm_outname_ext, load_gdal_dataset, save_gdal_dataset
from Masking import create_mask
from GeoFormatting import DS_to_extent
from SpatialStatistics import Variogram
from Viewing import plot_raster


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

    AnalysisArgs = parser.add_argument_group('ANALYSIS')
    AnalysisArgs.add_argument('-s','--max-samples', dest='maxSamples', type=int, default=1000,
        help='Maximum number of sample points. (0, ..., [1000], ...).')
    AnalysisArgs.add_argument('-l','--lag-spacing', dest='lagSpacing', type=int, default=None,
        help='Distance (in pixels) between lags. ([10^-2 * min dimension]).')
    AnalysisArgs.add_argument('-d','--max-lag', dest='maxLag', type=int, default=None,
        help='Distance (in pixels) for the maximum lag. ([2/3 maximum image dimension]).')
    AnalysisArgs.add_argument('-f','--fit-type', dest='fitType', type=str, default='exponential',
        help='Model type used to fit semi(co)variogram. ([exponential]).')

    OutputArgs = parser.add_argument_group('OUTPUTS')
    OutputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true',
        help='Verbose mode.')
    OutputArgs.add_argument('--plot-inputs', dest='plotInputs', action='store_true',
        help='Plot input maps and sample points.')
    OutputArgs.add_argument('-p','--plot', dest='plot', action='store_true',
        help='Plot results.')
    OutputArgs.add_argument('-o','--outname', dest='outName', type=str, default='Filt',
        help='Filtered map name.')

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
    mask = create_mask(img, inps.maskArgs)


    ## Compute variogram
    vgram = Variogram(img, mask=mask,
        maxSamples=inps.maxSamples,
        lagSpacing=inps.lagSpacing, maxLag=inps.maxLag,
        fitType=inps.fitType,
        verbose=inps.verbose)


    ## Plot image
    if inps.plotInputs: vgram.plot_inputs(img, mask)
    if inps.plot: vgram.plot_variogram()

    plt.show()