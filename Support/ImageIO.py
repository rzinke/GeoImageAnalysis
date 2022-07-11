'''
SHORT DESCRIPTION
Data set loading and saving, especially for GDAL compatibility.

FUTURE IMPROVEMENTS

TESTING STATUS
Tested.
'''

### IMPORT MODULES ---
import os
import re
from glob import glob
import numpy as np
from osgeo import gdal



### MISCELLANEOUS ---
def check_exists(fname):
    ''' Check that a file exists. '''
    assert os.path.exists(fname), 'ERROR: {:s} does not exist!!'.format(fname)


def pick_dataset(datasets, verbose=False):
    ''' Return the first entry in a dictionary. '''
    # Create list of datasets
    dsNames = list(datasets.keys())

    # Return first dataset
    dsName = dsNames[0]

    # Report if requested
    if verbose == True: print('Returning dictionary entry: {:s}'.format(dsName))

    # Return first data set
    return datasets[dsName]



### NAME FORMATTING ---
def rel_to_abs_paths(fnames):
    ''' Convert relative file paths to absolute file paths. '''
    # Use list if not already that type
    if type(fnames) != list: fnames = [fnames]

    # Convert relative paths to absolute paths
    fnames = [os.path.abspath(fname) for fname in fnames]

    return fnames


def confirm_outdir(fname):
    ''' Confirm that the output folder exists. '''
    # Check that output name is an absolute file path
    fname = os.path.abspath(fname)

    # Get directory
    dirName = os.path.dirname(fname)

    # Confirm directory exists
    if not os.path.exists(dirName):
        os.mkdir(dirName)


def append_fname(fname, appendix, verbose=False):
    '''
    "Append" a filename by placing the appendix after the main file name and
     before the extension.
    '''
    parts = fname.split('.')  # split extension
    ext = parts[-1]  # extension
    parts = '.'.join(parts[:-1])  # rejoin non-extension parts
    parts += appendix  # add appendix
    appName = '.'.join([parts, ext])  # reform with extension

    if verbose == True: print('Filename {:s} -> {:s}'.format(fname, parts))

    return appName


def confirm_outname_ext(outName, ext, verbose=False):
    '''
    Check that the output name uses the specified extension(s). If not,
     add that extension.
    '''
    # Strip off excess periods from extensions
    ext = [ex.strip('.') for ex in ext]

    # Check that extension is used
    if outName.split('.')[-1] not in ext:
        outName = '.'.join([outName, ext[0]])

    # Report if reequested
    if verbose == True: print('Out name: {:s}'.format(outName))

    return outName


def parse_extension(fname, verbose=False):
    ''' Retrieve the filename exension. '''
    # Get extension
    ext = fname.split('.')[-1]

    # Report if requested
    if verbose == True: print('Filename extension: {:s}'.format(ext))

    return ext


def confirm_overwrite(fname):
    ''' Check if file already exists and return True to confirm overwrite. '''
    # Check if file already exists
    if os.path.exists(fname):
        # Check file existence
        response = input('File already exists. Overwrite? (y/n) ')

        # Record user response
        if response.lower() in ['y', 'yes']:
            write_file = True  # overwrite
        else:
            write_file = False  # don't overwrite
    else:
        # File does not already exist
        write_file = True  # write new file

    return write_file



### LOADING GDAL DATASETS ---
def load_gdal_dataset(dsPath, verbose=False):
    ''' Load data set using GDAL. '''
    if verbose == True: print('Loading data set')

    # Check that data set path exists
    check_exists(dsPath)

    # Open GDAL data set
    DS = gdal.Open(dsPath, gdal.GA_ReadOnly)

    # Report if requested
    if verbose == True: print('Loaded: {:s}'.format(dsPath))

    return DS


def load_gdal_datasets(dsPaths, dsNames=None, verbose=False):
    '''
    Provide a list of dataset names.
    Check that the files actually exist, then load into dictionary.
    Returns a dictionary.
    '''
    if verbose == True: print('Loading multiple GDAL data sets')

    # Create list of data set names if not already provided
    if dsNames is None:
        dsNames = []  # empty list of names
        for i, dsPath in enumerate(dsPaths):
            # Formulate name
            dsName = os.path.basename(dsPath)  # path basename
            dsName = dsName.split('.')[:-1]  # remove extension
            dsName = '.'.join(dsName)  # reformulate name

            # Check that name not already in list
            if dsName in dsNames: dsName += '_{:d}'.format(i)

            dsNames.append(dsName)

    # Setup
    datasets = {}

    # Loop through data set names
    for i, dsPath in enumerate(dsPaths):
        # Data set name
        dsName = dsNames[i]

        # Report if requested
        if verbose == True: print('Loading {:s}: {:s}'.format(dsName, dsPath))

        # Load data set
        datasets[dsName] = load_gdal_dataset(dsPath)

    return datasets


def images_from_dataset(dataset, bands='all', verbose=False):
    '''
    Retrieve the specified bands from a GDAL data set.
    Bands can be specified as all or a list of integers starting at 1.
    '''
    # Determine number of bands
    if bands == 'all':
        nBands = dataset.RasterCount
        bands = range(1, nBands+1)
    else:
        bands = [int(band) for band in bands]
        nBands = len(bands)

    # Load image value arrays and flatten
    imgs = [dataset.GetRasterBand(band).ReadAsArray() for band in bands]

    # Report if requested
    if verbose == True:
        print('{:d} bands loaded'.format(nBands))

    return imgs


def images_from_datasets(datasets, band=1, verbose=False):
    ''' Retrieve the images from a list or dictionary of GDAL data sets. '''
    if verbose == True: print('Retrieving images from data sets')

    # Setup
    imgs = {}

    # Loop through data sets
    for dsName in datasets.keys():
        imgs[dsName] = datasets[dsName].GetRasterBand(band).ReadAsArray()

    return imgs



### SAVING GDAL DATASETS ---
def save_gdal_dataset(outName, imgs, mask=None, exDS=None,
                      proj='', tnsf=(0.0, 1.0, 0.0, 0.0, 0.0,- 1.0),
                      fmt='GTiff',
                      verbose=False):
    '''
    Save image to georeferenced data set, given an example data set.

    INPUTS
        ourName - data set output name
        imgs - single image or list of images; if a list is provided,
         each item in the list will be written to a separate band in the
         order in which they occcur
        (mask) is the mask to be applied to all bands. Masked values set to zero.
        (exDS) provides the projection and geotransform unless they are explicitly
         specified
        (proj) is the geographic projection, overrides exDS
        (tnsf) is the geographic transform, overrides exDS
        (fmt) is the output image format
    '''
    # Spatial parameters
    if proj == 'WGS84':
        proj = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'

    # Use existing data set properties
    if exDS is not None:
        # Overwrite projection and transform
        proj = exDS.GetProjection()
        tnsf = exDS.GetGeoTransform()

    # Convert single band to list if not already
    if type(imgs) != list:
        imgs = [imgs]

    # Image parameters
    nBands = len(imgs)

    # Check that all images are the same size
    M, N = imgs[0].shape
    for img in imgs:
        assert img.shape == (M, N), 'Images must be the same size'

    # Mask image(s) if mask is provided
    if mask is not None: 
        for img in imgs:
            img[mask==0] = 0  # apply mask value of zero

    # Establish driver
    driver = gdal.GetDriverByName(fmt)

    # Create data set
    DSout = driver.Create(outName, N, M, nBands, gdal.GDT_Float32)
    DSout.SetProjection(proj)
    DSout.SetGeoTransform(tnsf)

    # Write image bands
    for n in range(nBands):
        DSout.GetRasterBand(n+1).WriteArray(imgs[n])

    # Flush cache
    DSout.FlushCache

    # Report if requested
    if verbose == True: print('Saved raster to: {:s}'.format(outName))



### GPS LOADING ---
class GPSdata:
    def __init__(self, verbose=False):
        ''' Load, store, and format GPS data from text file. '''
        # Parameters
        self.verbose = verbose

    # Load data
    def load_from_file(self, fname,
        lonCol=0, latCol=1,
        Ecol=None, Ncol=None, Vcol=None, DATAcol=None,
        headerRows=0, delimiter=' '):
        ''' Load data from file. '''
        if self.verbose == True: print('Loading GPS from file')

        # Check file exists
        check_exists(fname)

        # Load GPS data from file
        with open(fname, 'r') as gpsData:
            # Read data and ignore header
            lines = gpsData.readlines()
            stations = lines[headerRows:]

        # Reformat data list
        stations = [station.strip('\n').split(delimiter) for station in stations]

        # Parse data
        Lon, Lat = [], []
        E, N, V, data = [], [], [], []

        # Parse coordinates
        for station in stations:
            Lon.append(station[lonCol])
            Lat.append(station[latCol])

        # Format into numpy arrays
        self.lon = np.array(Lon, dtype=float)
        self.lat = np.array(Lat, dtype=float)

        # Empty displacement attributes
        self.E = None
        self.N = None
        self.V = None
        self.data = None

        # Parse displacements
        if Ecol is not None:
            E = [station[Ecol] for station in stations]
            self.E = np.array(E, dtype=float)

        if Ncol is not None:
            N = [station[Ncol] for station in stations]
            self.N = np.array(N, dtype=float)

        if Vcol is not None:
            V = [station[Vcol] for station in stations]
            self.V = np.array(V, dtype=float)

        if DATAcol is not None:
            data = [station[DATAcol] for station in stations]
            self.data = np.array(data, dtype=float)

    def assign_zero_vertical(self):
        ''' Assign a value of zero for each displacement measurement. '''
        if self.verbose == True: print('Setting vertical values to 0')

        # Check that same number of E and N measurements are provided
        assert len(self.E) == len(self.N) > 0, 'East and North columns must be specified'

        # Assign zeros
        self.V = np.zeros(self.E.size)

    def scale_by_factor(self, scaleFactor):
        ''' Multiply all data values by the specified scale factor. '''
        if self.verbose == True: print('Scaling by factor: {:f}'.format(scaleFactor))

        # Scale data
        if self.E is not None: self.E *= scaleFactor
        if self.N is not None: self.N *= scaleFactor
        if self.V is not None: self.V *= scaleFactor
        if self.data is not None: self.data *= scaleFactor

    def compute_norms(self):
        ''' Compute the norm of the GPS components. '''
        if self.verbose == True: print('Computing GPS displacement magnitudes')

        # Ensure E, N, and V are all specified
        assert self.E is not None, 'East component must be specified'
        assert self.N is not None, 'North component must be specified'

        if self.V is None:
            self.assign_zero_vertical()

        # Find magnitude of displacement
        self.M = np.sqrt(self.E**2 + self.N**2 + self.V**2)

    def crop_to_extent(self, west=None, east=None, south=None, north=None):
        ''' Crop the data set to the given bounds. '''
        # Determine max bounds of data set
        if west is None: west = self.lon.min()
        if east is None: east = self.lon.max()
        if south is None: south = self.lat.min()
        if north is None: north = self.lat.max()

        # Indices of stations to include
        ndx = (self.lon >= west) & (self.lon <= east) & (self.lat >= south) & (self.lat <= north)

        # Crop data set
        self.crop_by_index(ndx)

        # Report if requested
        if self.verbose == True:
            print('Cropped data set to W {:f}, E {:f}, S {:f}, N {:f}'.format(west, east, south, north))

    def crop_by_index(self, ndx):
        ''' Keep only the values marked "True" in the ndx boolean array. '''
        # Parameters
        origLen = len(self.lon)

        assert len(ndx) == origLen, \
            'Index length ({:d}) must equal the number of stations ({:d})'.format(len(ndx), origLen)

        # Crop data set
        self.lon = self.lon[ndx]
        self.lat = self.lat[ndx]
        if self.E is not None: self.E = self.E[ndx]
        if self.N is not None: self.N = self.N[ndx]
        if self.V is not None: self.V = self.V[ndx]
        if self.data is not None: self.data = self.data[ndx]

        # Report if requested
        if self. verbose == True:
            print('Cropped by index. {:d}/{:d} values remain.'.format(len(self.lon), origLen))



### RELAX LOADING ---
def detect_relax_files(relaxDir, verbose=False):
    ''' Find all the Relax displacement files in a folder. '''
    if verbose == True: print('Detecting Relax filenames')

    # Basic strings
    Estr = '-east.grd'
    Nstr = '-north.grd'
    Ustr = '-up.grd'

    # Search expressions for different components
    srchE = '*{:s}'.format(Estr)
    srchN = '*{:s}'.format(Nstr)
    srchU = '*{:s}'.format(Ustr)

    # Search strings for different components
    srchStrE = os.path.join(relaxDir, srchE)
    srchStrN = os.path.join(relaxDir, srchN)
    srchStrU = os.path.join(relaxDir, srchU)

    # Search for filenames for different components
    Eresults = glob(srchStrE)
    Nresults = glob(srchStrN)
    Uresults = glob(srchStrU)

    # Find timestep numbers
    Enbs = list(set([re.findall('[0-9]{3}', os.path.basename(Nresult))[0] for Nresult in Nresults]))
    Nnbs = list(set([re.findall('[0-9]{3}', os.path.basename(Nresult))[0] for Nresult in Nresults]))
    Unbs = list(set([re.findall('[0-9]{3}', os.path.basename(Uresult))[0] for Uresult in Uresults]))

    # Sort numbers
    Enbs.sort()
    Nnbs.sort()
    Unbs.sort()

    # Check that all data are available
    assert Enbs == Nnbs == Unbs, 'E, N, and Up must have the same number of components'
    components = Enbs
    nComponents = len(components)

    # Report if requested
    if verbose == True: print('Number of components detected: {:d}'.format(nComponents))

    # Formulate file names
    Enames = ['{:s}{:s}'.format(Enb, Estr) for Enb in Enbs]
    Nnames = ['{:s}{:s}'.format(Nnb, Nstr) for Nnb in Nnbs]
    Unames = ['{:s}{:s}'.format(Unb, Ustr) for Unb in Unbs]

    # Formulate path names
    Epaths = [os.path.join(relaxDir, Ename) for Ename in Enames]
    Npaths = [os.path.join(relaxDir, Nname) for Nname in Nnames]
    Upaths = [os.path.join(relaxDir, Uname) for Uname in Unames]

    return Epaths, Npaths, Upaths



### PROFILE LOADING ---
def load_polyline(polylineName, verbose=False):
    ''' Load a polyline from a text file in x, y format. '''
    if verbose == True: print('Loading polyline file...')

    # Check file exists
    check_exists(polylineName)

    # Load file contents
    with open(polylineName, 'r') as polyFile:
        lines = polyFile.readlines()  # read all lines
        lines = lines[1:]  # exclude header

    x = []; y = []
    for line in lines:
        # Split by comma
        data = line.split(',')

        # Parse contents
        x.append(float(data[0]))
        y.append(float(data[1]))

    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # Number of vertices
    nVertices = len(x)

    # Report if requested
    if verbose == True:
        print('{:d} vertices detected'.format(nVertices))

    return x, y


class load_profile_data:
    def __init__(self, profFilename, verbose=False):
        ''' Load data from a saved profile. '''
        if verbose == True: print('Loading profile: {:s}'.format(profFilename))

        # Load profile contents
        with open(profFilename, 'r') as profFile:
            lines = profFile.readlines()

        # Parse profile data
        startXY = lines[0].replace('start: ', '').split(',')
        endXY = lines[1].replace('end: ', '').split(',')

        self.xStart = float(startXY[0])
        self.yStart = float(startXY[1])
        self.xEnd = float(endXY[0])
        self.yEnd = float(endXY[1])

        header = lines[2]
        profData = lines[3:]

        # Loop through data points
        nPts = len(profData)
        self.profDists = []; self.profPts = []

        for i in range(nPts):
            datum = profData[i].split(' ')
            self.profDists.append(float(datum[0]))
            self.profPts.append(float(datum[1]))

        # Convert to numpy arrays
        self.profDists = np.array(self.profDists)
        self.profPts = np.array(self.profPts)

def load_profile_endpoints(queryFilename, verbose=False):
    '''
    Load the profile endpoints from a text file. Coordinate should be in format
     startX,startY endX,endY
    '''
    if verbose == True: print('Loading profile endpoints')

    # Read file contents
    with open(queryFilename, 'r') as queryFile:
        profData = queryFile.readlines()

    # Parse coordinates
    startLons = []; startLats = []
    endLons = []; endLats = []

    for prof in profData:
        # Start and end points
        startPt, endPt = prof.split(' ')

        # Lon/lat
        startLon, startLat = startPt.split(',')
        startLons.append(float(startLon))
        startLats.append(float(startLat))

        endLon, endLat = endPt.split(',')
        endLons.append(float(endLon))
        endLats.append(float(endLat))

    # Number of profiles
    nProfs = len(startLons)

    # Report if requested
    if verbose == True: print('... {:d} profiles detected'.format(nProfs))

    return startLons, startLats, endLons, endLats



### PROFILE SAVING ---
def save_profile_data(outname, profStart, profEnd, profDist, profPts,
    verbose=False):
    ''' Save profile using standardized format. '''
    # Setup
    assert len(profDist) == len(profPts), \
        'Number of distance and measurements points must be identical'
    nPts = len(profDist)

    # File formatting
    metadata = 'start: {:f},{:f}\nend: {:f},{:f}\n'
    header = '# distance amplitude\n'
    dataStr = '{:f} {:f}\n'

    # Write contents to file
    with open(outname, 'w') as profFile:
        profFile.write(metadata.format(*profStart, *profEnd))
        profFile.write(header)
        for i in range(nPts):
            profFile.write(dataStr.format(profDist[i], profPts[i]))

    # Report if requested
    if verbose == True:
        print('Saved profile: {:s}'.format(outName))
