'''
SHORT DESCRIPTION
Image frequency analysis

FUTURE IMPROVEMENTS

TESTING STATUS
In development.
'''

### IMPORT MODULES ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from osgeo import gdal

from ImageIO import load_gdal_dataset
from GeoFormatting import DS_to_extent
from ImageMasking import create_mask
from ImageViewing import plot_raster



### POWER SPECTRAL DENSITY ---
class ImagePSD:
    def __init__(self, verbose=False):
        ''' Compute the power spectral density of an image. '''
        # Parameters
        self.verbose = verbose

    def load_image(self, imgName, bandNb=1):
        '''
        Load an image file as a GDAL data set.
        Otherwise, specify manually.
        '''
        # Load image
        DS = load_gdal_dataset(imgName, verbose=self.verbose)
        self.img = DS.GetRasterBand(bandNb).ReadAsArray()

        # Image dimensions
        self.M = DS.RasterYSize
        self.N = DS.RasterXSize

        del DS


    def compute_psd(self):
        '''
        Compute the power spectral density first using Fourier decomposition.
        '''
        if self.verbose == True:
            print('Compute PSD')

        # Check that image has been loaded
        assert hasattr(self, 'img'), 'Image must be loaded'

        # Enforce that the image is square
        self.__enforce_square__()

        # Fourier decomposition
        self.fourierImg = np.fft.fft2(self.img)
        self.fourierAmp = np.abs(self.fourierImg)

        # Construct wave vector array
        kfreq = np.fft.fftfreq(self.imgDim) * self.imgDim
        kfreq2D = np.meshgrid(kfreq, kfreq)

        # Wave vector distance from origin
        knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)  # L2-norm
        knrm = knrm.flatten()

        # Bins
        kbins = np.arange(0.5, self.imgDim//2+1, 1.)  # bin starts, ends
        self.kvals = 0.5 * (kbins[1:] + kbins[:-1])  # bin centers

        # Bin amplitudes
        self.binAmps, _, _ = binned_statistic(knrm, self.fourierAmp.flatten(),
                statistic="mean",
                bins=kbins)

        # Scale by surface area of concentric discs
        self.binAmps *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    def __enforce_square__(self):
        '''
        Check that image dimensions are equal.
        If they are not, pad the image.
        '''
        if self.M == self.N:
            # Report if requested
            if self.verbose == True:
                print('Image dimensions equal: {:d} x {:d}'.\
                    format(self.M, self.N))

        else:
            # Report if requested
            if self.verbose == True:
                print('Image dimensions: {:d} x {:d}\n... Padding'.\
                    format(self.M, self.N))

            # Pad image
            if self.M > self.N:
                self.img = np.pad(self.img,
                        ((0, 0), (0, self.M-self.N)),
                        'constant')

            elif self.N > self.M:
                self.img = np.pad(self.img,
                        ((0, self.N-self.M), (0, 0)),
                        'constant')

        # Image dimension
        self.imgDim = self.img.shape[0]

        # Report if requested
        if self.verbose == True:
            print('Square image dimension: {:d}'.format(self.imgDim))


    def plot(self, cmap='viridis', minPct=0, maxPct=100):
        '''
        Plot the original image, squared Fourier transform, and power spectral
         density.
        '''
        if self.verbose == True:
            print('Plotting')

        # Spawn figure and axes
        fig, [axImg, axFT, axPSD] = plt.subplots(ncols=3, figsize=(10,4))

        # Plot original image
        plot_raster(self.img,
            cmap=cmap, minPct=minPct, maxPct=maxPct,
            fig=fig, ax=axImg)

        # Format original image plot
        axImg.set_xticks([])
        axImg.set_yticks([])
        axImg.set_title('Orig image (padded)')

        # Format Fourier amplitude image for plotting
        fourierAmpPlot = np.fft.fftshift(self.fourierAmp)
        fourierAmpPlot = np.log10(fourierAmpPlot)

        # Plot Fourier series
        plot_raster(fourierAmpPlot,
            fig=fig, ax=axFT)

        # Format Fourier series plot
        axFT.set_xticks([])
        axFT.set_yticks([])
        axFT.set_title('log10(FT)')

        # Plot PSD
        axPSD.loglog(self.kvals, self.binAmps, 'k')

        # Format PSD plot
        axPSD.set_xlabel('k')
        axPSD.set_ylabel('P(k)')
        axPSD.set_title('Power spectral density')

        # Format figure
        fig.tight_layout()
