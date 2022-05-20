#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Examine the power spectral density of an image.

FUTURE IMPROVEMENTS

TESTING STATUS
In development.
'''

### IMPORT MODULES ---
import argparse
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from ImageFrequencyAnalysis import ImagePSD


### PARSER ---
Description = '''Examine the power spectral density of an image.'''

Examples = ''''''


def createParser():
    parser = argparse.ArgumentParser(description=Description,
        formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    InputArgs = parser.add_argument_group('INPUTS')
    InputArgs.add_argument(dest='imgName',
        help='Name of file to display.')
    InputArgs.add_argument('-b','--band', dest='bandNb', type=int, default=1,
        help='Image band number to display.')


    DisplayArgs = parser.add_argument_group('DISPLAY PARAMS')
    DisplayArgs.add_argument('-c','--cmap', dest='cmap', type=str, default='viridis',
        help='Colormap ([viridis]).')
    DisplayArgs.add_argument('-minPct','--min-percent', dest='minPct', type=float, default=None,
        help='Minimum percent clip value ([None]).')
    DisplayArgs.add_argument('-maxPct','--max-percent', dest='maxPct', type=float, default=None,
        help='Maximum percent clip value ([None]).')


    AnalysisArgs = parser.add_argument_group('ANALYSIS')


    OutputArgs = parser.add_argument_group('OUTPUTS')
    OutputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true',
        help='Verbose mode.')

    return parser


def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)



### Spectral analysis ---
if __name__ == '__main__':
    ## Inputs
    # Gather arguments
    inps = cmdParser()


    ## Power spectral density analysis
    # Initialize object
    psd = ImagePSD(verbose=inps.verbose)

    # Load image
    psd.load_image(imgName=inps.imgName,
        bandNb=inps.bandNb)

    # Compute power spectrum
    psd.compute_psd()

    # Plot
    psd.plot(cmap=inps.cmap,
        minPct=inps.minPct,
        maxPct=inps.maxPct)


    plt.show()
