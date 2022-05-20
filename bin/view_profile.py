#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Display a 1D profile.

FUTURE IMPROVEMENTS

TESTING STATUS
Tested.
'''

### IMPORT MODULES ---
import argparse
import matplotlib.pyplot as plt

from ImageIO import load_profile_data, confirm_outdir, append_fname, confirm_outname_ext
from ImageFitting import fit_linear, fit_atan
from ImageProfiling import profile_binning
from ImageViewing import plot_profile


### PARSER ---
Description = '''Display a profile.'''

Examples = ''''''


def createParser():
    parser = argparse.ArgumentParser(description=Description,
        formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    InputArgs = parser.add_argument_group('INPUTS')
    InputArgs.add_argument(dest='profName', type=str,
        help='Profile name.')
    InputArgs.add_argument('--fit-type', dest='fitType', type=str, default=None,
        help='Fit function ([None], atan, linearN).')
    InputArgs.add_argument('--binning', dest='binning', action='store_true',
        help='Binning.')
    InputArgs.add_argument('--bin-width', dest='binWidth', type=float, default=None,
        help='Binning width.')
    InputArgs.add_argument('--bin-spacing', dest='binSpacing', type=float, default=None,
        help='Binning width.')


    OutputArgs = parser.add_argument_group('OUTPUTS')
    OutputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true', 
        help='Verbose mode.')
    OutputArgs.add_argument('-o','--outname', dest='outName', type=str, default=None, 
        help='Output name head. ([None]).')

    return parser


def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)



### SAVING ---
def save_figure(outName, fig, verbose=False):
    ''' Save a formatted figure to the appropriate file path and name. '''
    # Format outname
    outName = confirm_outname_ext(outName, ['.png'])

    # Save figure
    fig.savefig(outName, dpi=600)

    if verbose == True: print('Saved figure to {:s}'.format(outName))


def save_bins(outName, x, y, verbose=False):
    ''' If binning is carried out, save the bin points to a file. '''
    # Format save name
    outName = confirm_outname_ext(outName, ['.txt'])
    outName = append_fname(outName, '_bins')

    # Save points to file
    with open(outName, 'w') as outFile:
        outFile.write('# distance amplitude\n')
        for i in range(len(x)):
            outFile.write('{:.4f} {:.4f}\n'.format(x[i], y[i]))

    if verbose == True: print('Saved bins to {:s}'.format(outName))


def save_fit(outName, fitType, profDist, yfit, verbose=False):
    ''' If the data were fit, save the fit curve to a file. '''
    # Format save name
    outName = confirm_outname_ext(outName, ['.txt'])
    outName = append_fname(outName, '{:s}_fit'.format(fitType))

    # Save points to file
    with open(outName, 'w') as outFile:
        outFile.write('# distance amplitude\n')
        for i in range(len(x)):
            outFile.write('{:.4f} {:.4f}\n'.format(x[i], y[i]))

    if verbose == True: print('Save fit points to {:s}'.format(outName))    



### MAIN ---
if __name__ == '__main__':
    ## Inputs
    # Gather arguments
    inps = cmdParser()


    ## Load profile
    profile = load_profile_data(inps.profName, verbose=inps.verbose)


    ## Plot data
    # Spawn figure
    fig, ax = plt.subplots(figsize=(9, 5))

    # Determine color
    if inps.fitType is None and inps.binning is False:
        color = 'k'
    else:
        color = (0.5, 0.5, 0.5)

    # Plot profile
    ax.scatter(profile.profDists, profile.profPts, s=25, c=[color], label='data')

    # Binning
    if inps.binning == True:
        x, y = profile_binning(profile.profDists, profile.profPts,
            binWidth=inps.binWidth, binSpacing=inps.binSpacing)
        ax.plot(x, y, color='b', linewidth=3,
            label='binning')

    # Fit profile if requested
    if inps.fitType is not None:
        if inps.fitType == 'atan':
            yfit, B = fit_atan(profile.profDists, profile.profPts, verbose=inps.verbose)

        elif inps.fitType[:6] == 'linear':
            fitDegree = int(inps.fitType[6:])
            yfit, B = fit_linear(profile.profDists, profile.profPts, fitDegree, verbose=inps.verbose)

        ax.plot(profile.profDists, yfit, color='r', linewidth=3,
            label='{:s} fit'.format(inps.fitType))

    # Format plot
    ax.set_xlabel('distance from start')
    ax.legend()

    # Save if outname provided
    if inps.outName:
        # Confirm output directory exists
        confirm_outdir(inps.outName)

        # Save figure
        save_figure(inps.outName, fig, verbose=inps.verbose)

        # Save bins if available
        if inps.binning == True: save_bins(inps.outName, x, y, verbose=inps.verbose)

        # Save profile fit if available
        if inps.fitType: save_fit(inps.outName, inps.fitType, profDist, yfit, verbose=inps.verbose)


    plt.show()