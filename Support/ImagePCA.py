'''
SHORT DESCRIPTION
Conduct a principal component analysis of the specified data set.

FUTURE IMPROVEMENTS
    Documentation

TESTING STATUS
In development.
'''

### IMPORT MODULES ---
import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

from ImageIO import confirm_outdir, confirm_outname_ext, append_fname, save_gdal_dataset
from ImageMasking import create_mask
from ImageViewing import dataset_clip_values, plot_raster, raster_multiplot



### PCA CLASS ---
class StackPCA:
    def __init__(self, verbose=False):
        '''
        Conduct a principal component analysis of the given images.
        The workflow is to stack the images.
        '''
        # Parameters
        self.verbose = verbose

    def load_images(self, imgList, maskArgs=None):
        '''
        Load the images from the list provided.
        Ensure the images are all in the same frame of reference.

        If only one image is provided, assume it is multi-band and load all
         bands.

        INPUTS
            imageList is a list of file names to be loaded
        '''
        # Number of images
        nImgs = len(imgList)
        if self.verbose == True:
            print('Loading {:d} images'.format(nImgs))

        # Empty variables
        self.imgs = []

        # Load initial data set for reference
        DS0 = gdal.Open(imgList[0], gdal.GA_ReadOnly)
        if self.verbose == True: print('\t{:s}'.format(imgList[0]))

        # Parse initial data set
        self.imgs.append(DS0.GetRasterBand(1).ReadAsArray())  # image map
        self.proj = DS0.GetProjection()
        self.tnsf = DS0.GetGeoTransform()
        self.M, self.N = DS0.RasterYSize, DS0.RasterXSize

        # Respond based on number of images provided
        if nImgs == 1:
            # Check if multi-band image
            assert DS0.RasterCount > 1, \
                'If single image is provided, must be multi-band'

            # Loop through bands
            for i in range(1, DS0.RasterCount+1):
                self.imgs.append(DS0.GetRasterBand(i).ReadAsArray())

            # Number of images
            self.K = DS0.RasterCount

        elif nImgs > 1:
            # Loop through images
            for imgName in imgList[1:]:
                if self.verbose == True: print('\t{:s}'.format(imgName))

                # Load data set
                DS = gdal.Open(imgName, gdal.GA_ReadOnly)
                self.imgs.append(DS.GetRasterBand(1).ReadAsArray())

                # Check that map bounds and sampling match the original
                assert DS.GetProjection() == self.proj, \
                    '{:s} projection is not the same as the original'.\
                    format(imgName)
                assert DS.GetGeoTransform() == self.tnsf, \
                    '{:s} geotransform is not the same as the original'.\
                    format(imgName)
                assert DS.RasterYSize == self.M and DS.RasterXSize == self.N, \
                    '{:s} size is not the same as the original'.format(imgName)

            # Number of images
            self.K = nImgs

        # Create mask
        self.__create_mask__(maskArgs)

        # Clear memory
        del DS0

    def set_images(self, imgs, maskArgs=None, proj='', tnsf=''):
        '''
        Provide images as a list or numpy array, rather than loading them
         from files.
        Check the image consistency and define the necessary parameters
         if provided as a list.

        Set the geographic parameters of the data set.
        If the images are loaded using `load_images`, the geographic
         information will already be available and this step will not
         be necessary.
        '''
        if self.verbose == True:
            print('Setting images')

        # Check whether images provided as list or numpy array
        if type(imgs) == np.ndarray:
            # Already in array format, ascribe to object directly
            self.imgs = imgs

            # Image sizes
            self.K, self.M, self.N = self.imgs.shape

        elif type(imgs) == list:
            # Number of images
            self.K = len(imgs)

            # Initial image size
            self.M, self.N = imgs[0].shape

            # Check other image sizes
            for img in imgs[1:]:
                assert img.shape == (self.M, self.N), 'Image sizes not consistent'

            # Format list as numpy array
            self.imgs = np.array(imgs)

        # Report if requested
        if self.verbose == True:
            print('\t{:d} images provided'.format(self.K))

        # Create mask
        self.__create_mask__(maskArgs)

        # Set geographic parameters
        self.proj = proj
        self.tnsf = tnsf

    def __create_mask__(self, maskArgs):
        ''' Create mask. '''
        self.mask = create_mask(self.imgs[0], maskArgs=maskArgs,
                verbose=self.verbose)


    def run_pca(self, standardize=True):
        ''' Apply the principal component analysis to the data set. '''
        if self.verbose == True:
            print('Applying PCA to {:d} images'.format(self.K))

        # Reshape data into 2D (M.N x K) array
        data = np.zeros((self.M*self.N, self.K))
        for k in range(self.K):
            data[:,k] = self.imgs[k].flatten()

        if self.verbose == True:
            print('Formatted into {:d} x {:d} array'.format(*data.shape))

        # Reshape mask into 1D (M.N) array
        mask = self.mask.reshape(self.M*self.N, 1)

        # Standardize data by removing mean
        if standardize == True:
            if self.verbose == True:
                print('Standardizing data set')

            # Compute mean of each column
            means = np.mean(data, axis=0)

            # Subtract column means
            data = data - means

            # Mask values
            data = data * mask

        # Compute covariance matrix
        Cov = np.cov(data.T)

        # Eigen decomposition
        self.eigvals, self.eigvecs = np.linalg.eig(Cov)

        # Sort eigenvalues largest-smallest
        sortNdx = np.argsort(self.eigvals)[::-1]
        self.eigvals = self.eigvals[sortNdx]
        self.eigvecs = self.eigvecs[:,sortNdx]  # sort vectors by column

        # Relative importance of inputs
        self.inputImportance = np.argsort(np.abs(self.eigvecs[:,0]))[::-1]

        # Report if requested
        if self.verbose == True:
            print('Eigenvalues:\n{}'.format(self.eigvals))
            print('Relative importance of inputs (band nbs)\n{}'.\
                format(self.inputImportance+1))

        # Project data into eigen coordinates
        projData = np.dot(self.eigvecs.T, data.T)

        # Reshape data into image shapes
        self.PCs = np.zeros((self.K, self.M, self.N))
        for k in range(self.K):
            self.PCs[k,:,:] = projData[k,:].reshape(self.M, self.N)

    def recombine_pcs(self, nComponents):
        ''' Recombine the first n PCs to make a single image. '''
        if self.verbose == True:
            print('Combining first {:d} PCs'.format(nComponents))

        # Recombine
        comboImg = np.zeros((self.M, self.N))
        for i in range(nComponents):
            comboImg += self.eigvals[i]*self.PCs[i]

        return comboImg


    def plot_inputs(self, cmap='jet'):
        ''' Plot input images. '''
        # Determine color bounds
        vmin, vmax = dataset_clip_values(self.imgs, minPct=1, maxPct=99)

        # Plot images
        raster_multiplot(self.imgs, mrows=3, ncols=4,
            cmap=cmap, vmin=vmin, vmax=vmax)

    def plot_retention(self):
        ''' Plot the variance retained by each principal component. '''
        # Spawn figure and axis
        fig, ax = plt.subplots()

        # Plot eigenvalues
        ax.bar(range(self.K), self.eigvals/np.sum(self.eigvals)*100,
            align='center', width=0.4)

        # Format plot
        ax.set_xticks(range(self.K))
        ax.set_xticklabels(['PC{:d}'.format(k+1) for k in range(self.K)])

        ax.set_ylabel('% variance explained')

        ax.set_title('Variance retention')

    def plot_components(self, nComponents, cmap='jet'):
        ''' Plot the constituent components, one at a time for clarity. '''
        # Plot each component
        for i in range(nComponents):
            fig, ax = plot_raster(self.PCs[i,:,:], mask=self.mask,
                    cmap=cmap, cbarOrient='vertical')
            ax.set_title('Component {:d}'.format(i+1))

        # Plot recombined image
        comboImg = self.recombine_pcs(nComponents)
        fig, ax = plot_raster(comboImg, mask=self.mask,
                cmap=cmap, cbarOrient='vertical')
        ax.set_title('Combined {:d} PCs'.format(nComponents))

    def save(self, outname, nComponents='all'):
        ''' Save the PCs and recombined image to GDAL data sets. '''
        # Number of components
        if nComponents == 'all':
            nComponents = self.K

        # Checks
        confirm_outdir(outname)
        outname = confirm_outname_ext(outname, ext=['tif'])

        # Save names
        pcsName = append_fname(outname, '_{:d}PCs'.format(nComponents))
        comboName = append_fname(outname, '_{:d}recombined'.format(nComponents))

        # Save PC to data set
        save_gdal_dataset(pcsName,
            [self.PCs[i] for i in range(nComponents)],
            mask=self.mask, proj=self.proj, tnsf=self.tnsf,
            verbose=self.verbose)

        # Save recombined image to data set
        comboImg = self.recombine_pcs(nComponents)
        save_gdal_dataset(comboName, comboImg, mask=self.mask,
            proj=self.proj, tnsf=self.tnsf, verbose=self.verbose)
