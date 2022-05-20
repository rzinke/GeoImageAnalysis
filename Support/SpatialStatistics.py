'''
SHORT DESCRIPTION
Compute image spatial statistics.

FUTURE IMPROVEMENTS

TESTING STATUS
In development.
'''

### IMPORT MODULES ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from LinearFiltering import mean_filter
from ImageViewing import plot_raster



### STANDARD DEVIATION ---
def imgstdev(img, w=3, verbose=False):
    '''
    Compute the standard deviation for patches of pixels with the given width
     using a linear filter.
    '''
    if verbose == True: print('Computing standard deviation map')

    # Compute expectations
    EX = mean_filter(img, w)
    EXsq = mean_filter(img**2, w)

    # Compute variances
    var = EXsq - EX**2
    var[var<0] = 0  # correct rounding error

    # Compute standard deviations
    sd = np.sqrt(var)

    return sd



### VARIOGRAM ---
class SampleSet:
    def __init__(self, x, y):
        '''
        This object stores x and y points representing points on a map.
        INPUTS
         x, y are numpy arrays
        '''
        # Check input type is numpy array
        if type(x) != np.ndarray: x = np.array(x)
        if type(y) != np.ndarray: y = np.array(y)

        # Store values
        self.x = x
        self.y = y


    def clip_to_bounds(self, M, N):
        '''
        Clip the x, y values to the bounds of the area.
        INPUTS
         M, N are the y- and x-extents of the study area
        '''
        # Indices meeting conditions
        ndx = (self.x > 0) & (self.x < N) & (self.y > 0) & (self.y < M)

        # Filter samples within bounds
        self.x = self.x[ndx].astype(int)
        self.y = self.y[ndx].astype(int)


    def filter_by_mask(self, mask):
        '''
        Exclude samples of masked pixels.
        Make sure that the mask is the same size as the study area.
        '''
        # Indices of non-masked pixels
        ndx = (mask[self.y, self.x] != 0)

        # Filter samples within bounds
        self.x = self.x[ndx].astype(int)
        self.y = self.y[ndx].astype(int)


class Variogram:
    def __init__(self, img, mask=None,
        maxSamples=1000,
        lagSpacing=None, maxLag=None,
        fitType='exponential',
        verbose=False):
        '''
        Compute the semivariogram of an image data set.
        (Semi)variance will be computed at d distances from n sample points.

        INPUTS
         img is the image to be evaluated
         mask is an image mask (0s are masked)
         maxSamples is the number of samples to collect (before masking)
         lagSpacing is the distance (in pixels) between lags
         maxLag is the maximum distance from a point at which the image will be
          sampled
         fitType is the model type used to fit the semi(co)variogram

        The workflow consists of two parts:
         First, sample the image to compute the semivariances, semicovariances,
         and absolute differences.
         Second, organize the data and fit statistical models to it. To carry
         out this second stage, invoke a second object called VariogramStats.
        '''
        # Record parameters
        self.verbose = verbose

        # Report if requested
        if self.verbose == True: print('Computing image structure')

        # Format mask
        mask = self.__format_mask__(img, mask)

        # Sample image
        self.__sample_image__(img, mask, maxSamples, lagSpacing, maxLag)

        # Compute variances
        self.__compute_variances__(img)

        # Analyze variances
        self.analyses = VariogramStats(self.lagSpacing, self.lagDists,
            self.semivariances, self.semicovariances, self.absdifferences,
            fitType,
            verbose=self.verbose)

    def __format_mask__(self, img, mask):
        '''
        Check that a mask is provided. Create one signifying all valid values
         if not.
        '''
        if self.verbose == True: print('|\nFormatting mask')

        # Create mask if none provided
        if mask is None:
            mask = np.ones(img.shape)
        else:
            # Check that image and mask are equivalent sizes
            assert mask.shape == img.shape, \
                'Mask shape ({:d}x{:d} must equal image shape ({:d}x{:d})'.\
                format(mask.shape, img.shape)

        return mask

    def __sample_image__(self, img, mask, maxSamples, lagSpacing, maxLag):
        '''
        Sample the image at various lags for multiple points prior to variogram
         computation.
        INPUTS
         maxSamples is the number of samples to collect (before masking)
        '''
        if self.verbose == True: print('|\nSampling image')

        # Spatial parameters
        M, N = img.shape
        if self.verbose == True:
            print('Image size: {:d} x {:d}'.format(M, N))

        # Compute sample grid
        sx, sy = self.__compute_sample_grid__(mask, M, N, maxSamples)

        # Create query points
        px, py = self.__create_lag_points__(mask, M, N, lagSpacing, maxLag)

        # Associate query points with sample points
        self.__assemble_samples__(sx, sy, px, py, M, N, mask)

        # Summary
        self.nSamples = len(self.samples)
        self.lagsPerSample = [len(sample.x) for sample in self.samples]  # lag counts
        self.lagsPerSample = sum(self.lagsPerSample)/self.nSamples  # sum of lag counts/nb samples
        self.lagsPerSample = int(self.lagsPerSample)  # integer type

        # Report if requested
        if self.verbose == True:
            print('Summary: {:d} samples, with average {:d} lags per sample'.\
                format(self.nSamples, self.lagsPerSample))

    def __compute_sample_grid__(self, mask, M, N, P):
        '''
        Compute the grid of points for variogram construction.
        '''
        if self.verbose == True: print('Assembling sample grid')

        # Sample points on regular grid
        sDist = self.__compute_grid_spacing__(M, N, P)

        # Grid vectors
        sx = np.arange(0, N-sDist, sDist)
        sy = np.arange(0, M-sDist, sDist)

        # Grid
        sx, sy = np.meshgrid(sx, sy)

        # Stagger columns
        sx[::2, :] = sx[::2, :] + sDist/2

        # Reformat
        sx = sx.astype(int).flatten()
        sy = sy.astype(int).flatten()

        # Remove points in masked pixels
        ndx = (mask[sy,sx] == 1)  # non-masked sample indices
        sx = sx[ndx]  # valid sample x-locations
        sy = sy[ndx]  # valid sample y-locations

        # Report if requested
        if self.verbose == True:
            print('\t{:d} samples, spaced every {:.0f} pixels'.\
                format(sum(ndx), sDist))

        return sx, sy

    def __compute_grid_spacing__(self, M, N, P):
        '''
        Compute the sample spacing.
        INPUTS
         M, N are the image size (y, x)
         P is the maximum number of samples, before masking
        OUTPUTS
         sDist is the sample spacing
        '''
        return np.sqrt(M*N/P)

    def __create_lag_points__(self, mask, M, N, lagSpacing, maxLag):
        '''
        Find the points at which to compute the lags.
        Each sample point has a set of lags.
        By default, use 1000 points across the shortest axis of the image.
        '''
        if self.verbose == True: print('Assembling lags')

        # Determine lag distance
        self.__determine_lag_dist__(M, N, lagSpacing)

        # Determine maximum sampling distance across the image
        self.__determine_max_lag__(M, N, maxLag)

        # Lag distances (sample radii)
        dists = np.arange(1, self.maxLag, self.lagSpacing)

        # Lag angles (angles about which to sample)
        angles = np.arange(0, 2*np.pi, np.pi/15)

        # Determine query points
        px = np.outer(dists, np.cos(angles)).flatten()
        py = np.outer(dists, np.sin(angles)).flatten()

        # Report if requested
        if self.verbose == True:
            print('\tup to {:d} lag points per sample'.format(len(px)))

        return px, py

    def __determine_lag_dist__(self, M, N, lagSpacing):
        '''
        Determine or confirm the distance in pixels between lags.
        '''
        if not lagSpacing:
            # Minimum dimension
            minDim = min(M, N)

            # Automatically determine lag distance
            lagSpacing = minDim/100

        # Confirm lag distance formatting
        lagSpacing = int(round(lagSpacing))

        # Store to object
        self.lagSpacing = lagSpacing

        # Report if requested
        if self.verbose == True:
            print('\tdistance between lags: {:d} pixels'.format(self.lagSpacing))

    def __determine_max_lag__(self, M, N, maxLag):
        '''
        Determine the maximum sampling distance across the image.
        For now, use some fraction of the maximum image dimension.
        In the future, consider something more sophisticated.
        '''
        # Maximum dimension
        maxDim = max(M, N)

        # Determine maximum lag
        if not maxLag:
            # Compute maximum lag
            maxLag = 0.67 * maxDim

        # Confirm max lag formatting
        maxLag = int(maxLag)

        # Store to object
        self.maxLag = maxLag

        # Report if requested
        if self.verbose == True:
            print('\tmaximum lag: {:d} pixels'.format(self.maxLag))

    def __assemble_samples__(self, sx, sy, px, py, M, N, mask):
        '''
        Take the sample points and lags and build arrays of query points.
        Use the SampleSet class to associate x and y points.
        Filter the query points to include only those within the image area,
         and those sampling non-masked values.
        '''
        if self.verbose == True: print('Assembling sample points')

        # Setup
        nSamples = len(sx)

        # Create sample sets
        self.samples = [SampleSet(sx[i]+px, sy[i]+py) for i in range(nSamples)]

        # Retain only samples within image bounds
        [sample.clip_to_bounds(M, N) for sample in self.samples]

        # Retain on non-masked pixels
        [sample.filter_by_mask(mask) for sample in self.samples]

    def __compute_variances__(self, img):
        '''
        Once the image has been appropriately sampled, compute the semivariance,
         semicovariance, and semi-absolute difference.
        '''
        if self.verbose == True: print('|\nComputing variances')

        # Setup
        self.lagDists = np.array([])  # lag distances in pixels
        self.semivariances = np.array([])  # variances
        self.semicovariances = np.array([])  # covariances
        self.absdifferences = np.array([])  # absolute differences

        # Loop through samples
        for sample in self.samples:
            # Distances from sample center at each lag
            self.lagDists = np.concatenate([
                self.lagDists,
                np.sqrt((sample.x - sample.x[0])**2 + (sample.y - sample.y[0])**2)
                ])

            # Sample point from map
            sampleValues = img[sample.y, sample.x]

            # Compute semivariances
            self.semivariances = np.concatenate([
                self.semivariances, 
                0.5*(sampleValues - sampleValues[0])**2
                ])

            # Compute semicovariances
            self.semicovariances = np.concatenate([
                self.semicovariances,
                0.5*(sampleValues*sampleValues[0])
                ])

            # Compute absolute differences
            self.absdifferences = np.concatenate([
                self.absdifferences,
                np.abs(sampleValues - sampleValues[0])
                ])


    def plot_inputs(self, img, mask):
        '''
        Plot the sampling scheme used for the input image.
        '''
        if self.verbose == True: print('\nPlotting inputs')

        # Spawn figure and axis
        fig, axes = plt.subplots(figsize=(8, 4), ncols=2)

        cbarOrient = 'auto'

        # Plot image
        plot_raster(img, mask=mask,
            cbarOrient=cbarOrient, minPct=1, maxPct=99,
            fig=fig, ax=axes[0])
        axes[0].set_title('Input image')

        # Plot mask
        plot_raster(mask,
            cbarOrient=cbarOrient, fig=fig, ax=axes[1])
        axes[1].set_title('Image mask')
        axes[1].set_yticks([])

        # Plot sample points
        for i in range(len(axes)):
            [axes[i].scatter(sample.x[0], sample.y[0], 25, c='k', marker='+') \
                for sample in self.samples]


    def plot_variogram(self):
        '''
        Plot the variogram outputs.
        '''
        if self.verbose == True: print('\nPlotting variogram')

        ## Setup
        # Spawn figure and axis
        fig, axes = plt.subplots(figsize=(8, 6), nrows=3)

        ## Plot data
        # Plot semivariances
        axes[0].plot(self.analyses.lags, self.analyses.semivarMean,
            marker='o', linewidth=0, color='g')
        axes[0].errorbar(self.analyses.lags, self.analyses.semivarQuantiles[1,:],
            yerr=[self.analyses.semivarQuantiles[1,:] - self.analyses.semivarQuantiles[0,:],
                    self.analyses.semivarQuantiles[2,:] - self.analyses.semivarQuantiles[1,:]],
                    color='g', linestyle='none')

        # Plot semicovariances
        axes[1].plot(self.analyses.lags, self.analyses.semicovarMean,
            marker='o', linewidth=0, color='b')
        axes[1].errorbar(self.analyses.lags, self.analyses.semicovarQuantiles[1,:],
            yerr=[self.analyses.semicovarQuantiles[1,:] - self.analyses.semicovarQuantiles[0,:],
                    self.analyses.semicovarQuantiles[2,:] - self.analyses.semicovarQuantiles[1,:]],
                    color='b', linestyle='none')

        # Plot absolute differences
        axes[2].plot(self.analyses.lags, self.analyses.absdiffMean,
            marker='o', linewidth=0, color='r')
        axes[2].errorbar(self.analyses.lags, self.analyses.absdiffQuantiles[1,:],
            yerr=[self.analyses.absdiffQuantiles[1,:] - self.analyses.absdiffQuantiles[0,:],
                    self.analyses.absdiffQuantiles[2,:] - self.analyses.absdiffQuantiles[1,:]],
                    color='r', linestyle='none')

        ## Plot models
        # Plot semivariogram model
        axes[0].plot(self.analyses.lags,
            semivariogram_exponential_model(self.analyses.lags, *self.analyses.Vopt),
            color='gray', linewidth=2)  # plot model curve
        axes[0].plot([0, 0], [0, self.analyses.Vopt[2]],
            color='gray', linestyle='--',
            label='Nugget {:.4f}'.format(self.analyses.Vopt[2]))  # plot nugget
        axes[0].axvline(self.analyses.Vopt[0],
            color='gray', linestyle='--',
            label='Char dist {:.0f}'.format(self.analyses.Vopt[0]))  # plot range
        axes[0].axhline(self.analyses.Vopt[1],
            color='gray', linestyle='--',
            label='Sill {:.4f}'.format(self.analyses.Vopt[1]))  # plot sill

        # Plot semicovariogram model
        axes[1].plot(self.analyses.lags,
            semicovariogram_exponential_model(self.analyses.lags, *self.analyses.Copt),
            color='gray', linewidth=2)


        ## Formatting
        # Format titles and labels
        axes[0].set_ylabel('semi-\nvariance')
        axes[0].legend()
        axes[1].set_ylabel('semi-\ncovariance')
        axes[2].set_ylabel('abs.\ndifference')
        axes[2].set_xlabel('distance')

        # Format axes
        axes[0].set_xticks([])
        axes[1].set_xticks([])


class VariogramStats:
    def __init__(self, lagSpacing, lagDists,
        semivariances, semicovariances, absdifferences,
        fitType,
        verbose=False):
        '''
        Analyze the data produced by the Variogram class:
         Bin the lags
         Fit models to the semivariances and semicovariances
        '''
        # Record parameters
        self.verbose = verbose

        # Report if requested
        if self.verbose == True: print('\nAnalyzing variances')

        # Bin lag distances
        binnedLagDists = self.__bin_lags__(lagSpacing, lagDists)

        # Set up arrays
        self.__set_up_arrays__()

        # Compute statistics
        self.__compute_bin_stats__(binnedLagDists,
            semivariances, semicovariances, absdifferences)

        # Fit models
        self.__fit_models__(fitType)

    def __bin_lags__(self, lagSpacing, lagDists):
        '''
        Bin lags so each bin is characterized by a single value.
        '''
        if self.verbose == True: print('Binning lags')

        # Binned lags
        binnedLagDists = lagSpacing * np.round(lagDists/lagSpacing)

        # List of unique lag values
        lags = list(set(binnedLagDists))
        lags.sort()
        self.lags = np.array(lags)  # convert to array and store

        # Report if requested
        if self.verbose == True: print('\t{:d} lags'.format(len(self.lags)))

        return binnedLagDists

    def __set_up_arrays__(self):
        '''
        Set up arrays to store statistical computations.
        '''
        # Number of lags
        nLags = len(self.lags)

        # Semivariance
        self.semivarQuantiles = np.zeros((3, nLags))
        self.semivarMean = np.zeros((nLags))

        # Semicovariance
        self.semicovarQuantiles = np.zeros((3, nLags))
        self.semicovarMean = np.zeros((nLags))

        # Absolute difference
        self.absdiffQuantiles = np.zeros((3, nLags))
        self.absdiffMean = np.zeros((nLags))

    def __compute_bin_stats__(self, binnedLagDists,
        semivariances, semicovariances, absdifferences):
        '''
        Compute the statistics of each set of binned values:
         mean, and 10th and 90th percentiles
        '''
        if self.verbose == True: print('Computing statistics for each bin')

        # Establish quantiles of interest
        quantiles = (25, 50, 75)

        # Loop through bins
        for i, lag in enumerate(self.lags):
            # Indices of relevant data
            ndx = (binnedLagDists == lag)

            # Compute percentiles
            semivarQuantiles = np.percentile(semivariances[ndx], quantiles)

            semicovarQuantiles = np.percentile(semicovariances[ndx], quantiles)

            absdiffQuantiles = np.percentile(absdifferences[ndx], quantiles)

            # Append to arrays
            self.semivarQuantiles[:,i] = semivarQuantiles
            self.semicovarQuantiles[:,i] = semicovarQuantiles
            self.absdiffQuantiles[:,i] = absdiffQuantiles

            # Compute means
            self.semivarMean[i] = semivariances[ndx].mean()
            self.semicovarMean[i] = semicovariances[ndx].mean()
            self.absdiffMean[i] = absdifferences[ndx].mean()

    def __fit_models__(self, fitType):
        '''
        Analyze the variances computed.

        Bin the lags.
        Fit models to the semivariances and semicovariances.
        '''
        if self.verbose == True: print('Fitting model curves')

        # Setup
        fitType = fitType.lower()
        maxLag = self.lags.max()

        # Fit model based on fit type
        if fitType == 'exponential':
            # Fit semivariogram
            self.Vopt, self.Vcov = curve_fit(semivariogram_exponential_model,
                self.lags/maxLag,  # normalize lag values
                self.semivarMean)
            self.Vopt[0] *= maxLag  # re-scale lag values

            # Fit semicovariogram
            self.Copt, self.Ccov = curve_fit(semicovariogram_exponential_model,
                self.lags/maxLag,  # normalize lag values
                self.semicovarMean)
            self.Copt[0] *= maxLag  # re-scale lag values

        # Report stats if requested
        if self.verbose == True:
            print('Nugget: {:.4f}'.format(self.Vopt[2]))
            print('Characteristic distance: {:.1f}'.format(self.Vopt[0]))
            print('Sill: {:.4f}'.format(self.Vopt[1]))


def semivariogram_exponential_model(d, d0, A, B):
    '''
    Model curve to fit a semivariogram.
    Fvar = A (1-exp(-d/d0)) + B
    '''
    return A*(1-np.exp(-d/d0))+B


def semicovariogram_exponential_model(d, d0, C):
    '''
    Model curve to fit a semicovariogram.
    Fcov = C exp(-d/d0)
    '''
    return C*np.exp(-d/d0)
