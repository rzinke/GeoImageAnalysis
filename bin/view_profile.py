#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Display a 1D profile.

FUTURE IMPROVEMENTS

TESTING STATUS
Tested.
'''

### IMPORT MODULES ---
import os
import argparse
import matplotlib.pyplot as plt

from ImageIO import load_profile_data, save_profile_data, confirm_outdir, append_fname, confirm_outname_ext
from ImageFitting import fit_linear, fit_atan
from ImageProfiling import profile_binning


### PARSER ---
Description = '''Display a profile.'''

Examples = ''''''


def createParser():
    parser = argparse.ArgumentParser(description=Description,
        formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    InputArgs = parser.add_argument_group('INPUTS')
    InputArgs.add_argument(dest='profNames', type=str, nargs='+',
        help='Profile name.')
    InputArgs.add_argument('--fit-type', dest='fitType', type=str,
        choices=[None, 'atan', 'polynomial'], default=None,
        help='Fit function.')
    InputArgs.add_argument('--fit-degree', dest='fitDegree', type=int, default=1,
        help='Polynomial fit degree ([1], 2, 3, ...')
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

    # Confirm output directory exists if outname provided
    if inps.outName:
        confirm_outdir(inps.outName)


    ## Load profiles
    # Number of profiles to consider
    nProfiles = len(inps.profNames)

    # Report if requested
    if inps.verbose == True:
        print('{:d} profiles provided'.format(nProfiles))


    ## Plot data
    # Spawn figure
    fig, ax = plt.subplots(figsize=(9, 5))


    ## Plot profiles
    # Pre-define colors
    colors = ['b', 'r', 'g', 'm', 'c']
    nColors = len(colors)

    # Determine transparency
    if inps.fitType is None and inps.binning is False:
        # Plot only measured points
        profAlpha = 1.0
    else:
        # Use paler color for measured points
        profAlpha = 0.5

    # Plot profiles
    for i, profName in enumerate(inps.profNames):
        # Single-profile case
        if nProfiles == 1:
            profColor = 'k'
            binsColor = 'b'
            fitColor = 'r'
        elif nProfiles > 1:
            # Determine color scheme
            color = colors[i%nColors]
            profColor = color
            binsColor = color
            fitColor = color

        # Load profiles into profile objects
        profile = load_profile_data(profName, verbose=inps.verbose)

        # Plot profile
        ax.scatter(profile.profDists, profile.profPts,
            s=25, c=[profColor], alpha=profAlpha, edgecolors='none',
            label='data')

        # Binning
        if inps.binning == True:
            x, y = profile_binning(profile.profDists, profile.profPts,
                    binWidth=inps.binWidth, binSpacing=inps.binSpacing)
            ax.plot(x, y, color=binsColor, linewidth=3,
                label='binning')

        # Fit profile if requested
        if inps.fitType is not None:
            if inps.fitType == 'atan':
                yfit, B = fit_atan(profile.profDists, profile.profPts,
                        verbose=inps.verbose)

            elif inps.fitType == 'polynomial':
                yfit, B = fit_linear(profile.profDists, profile.profPts,
                        inps.fitDegree, verbose=inps.verbose)

            ax.plot(profile.profDists, yfit, color=fitColor, linewidth=3,
                label='{:s} fit'.format(inps.fitType))

        # Save if requested
        if inps.outName:
            # Base name of profile
            basename = os.path.basename(profName)
            basename = '.'.join(basename.split('.')[:-1])  # remove extension

            # Save bins if available
            if inps.binning:
                binsName = '{:s}_{:s}_bins.txt'.format(inps.outName, basename)
                save_profile_data(outname=binsName,
                    profStart=[profile.xStart, profile.yStart],
                    profEnd=[profile.xEnd, profile.yEnd],
                    profDist=x, profPts=y,
                    verbose=inps.verbose)

            # Save profile fit if available
            if inps.fitType:
                fitName = '{:s}_{:s}_fit.txt'.format(inps.outName, basename)
                save_profile_data(outname=fitName,
                    profStart=[profile.xStart, profile.yStart],
                    profEnd=[profile.xEnd, profile.yEnd],
                    profDist=profile.profDists, profPts=yfit,
                    verbose=inps.verbose)
                # save_fit(fitName, inps.fitType, profile.profDists, yfit,
                #     verbose=inps.verbose)


    # Format plot
    ax.set_xlabel('distance from start')
    ax.legend()

    # Save final figure if outname provided
    if inps.outName:
        # Save figure
        save_figure(inps.outName, fig, verbose=inps.verbose)


    plt.show()
