"""
Create composites of the raw data after removing the ensemble mean and then
calculating a rolling standard deviation

Reference  : Barnes et al. [2020, JAMES]
Author    : Zachary M. Labe
Date      : 17 January 2021
"""

### Import packages
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import scipy.stats as stats
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import palettable.scientific.sequential as ssss
import palettable.scientific.diverging as dddd
import cmocean as cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/'
directoryfigure = '/Users/zlabe/Desktop/ModelComparison/'
###############################################################################
###############################################################################
modelGCMs = ['CCCma_canesm2','MPI','CSIRO_MK3.6','KNMI_ecearth',
              'GFDL_CM3','GFDL_ESM2M','lens']
datasetsingle = ['SMILE']
seasons = ['annual']
variq = 'T2M'
reg_name = 'SMILEGlobe'
simuqq = datasetsingle[0]
monthlychoice = seasons[0]
lat_bounds,lon_bounds = UT.regions(reg_name)
###############################################################################
###############################################################################
land_only = False
ocean_only = False
rm_merid_mean = False
rm_annual_mean = False
rm_ensemble_mean = True
###############################################################################
###############################################################################
window = 5
if window == 0:
    rm_standard_dev = False
    yearsall = np.arange(1950,2019+1,1)
    ravel_modelens = False
    ravelmodeltime = True
else:
    rm_standard_dev = True
    yearsall = np.arange(1950+window,2019+1,1)
    ravelmodeltime = True
    ravel_modelens = True
###############################################################################
###############################################################################
numOfEns = 16
ensnum = numOfEns
lensalso = True
lentime = len(yearsall)
iterations = [100]
###############################################################################
###############################################################################
ravelyearsbinary = False
ravelbinary = False
num_of_class = len(modelGCMs)
###############################################################################
###############################################################################
lrpRule = 'z'
normLRP = True
###############################################################################
###############################################################################
### Create sample class labels for each model for my own testing
if seasons != 'none':
    lengthlabels = numOfEns * lentime #ensembles*years
    arrayintegers = np.arange(0,num_of_class,1)
    classesltest = np.repeat(arrayintegers,lengthlabels)

### Read in data
def readData(window,typemodel,variq,simuqq,land_only,reg_name,rm_standard_dev):
    """
    Read in LRP maps
    """
    
    data = Dataset(directorydata + name = 'LRP_Maps_ModelBiases-STDDEV%syrs_%s_Annual_%s_%s_land_only-%s_%s_STD-%s.nc' % (window,typemodel,variq,simuqq,land_only,reg_name,rm_standard_dev))
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    lrp = data.variables['LRP'][:]
    data.close()
    
    return lrp,lat,lon

### Function for calculating mean composites
def comp(data,yearcomp):
    """
    Calculate composites for first yearcomp and last yearcomp and take
    the difference
    """
    
    ### Take periods
    early = data[:,:yearcomp,:,:]
    late = data[:,-yearcomp:,:,:]
    
    ### Average periods
    earlym = np.nanmean(early,axis=1)
    latem = np.nanmean(late,axis=1)
    
    ### Difference
    diff = latem - earlym
    return diff

### Read in prediction data
trainindices = np.genfromtxt(directorydata + 'trainingEnsIndices_ModelBiases_%s_%s_%s_%s_iterations%s_STD-%s.txt' % (variq,monthlychoice,reg_name,dataset,iterations[0],rm_standard_dev))
testindices = np.genfromtxt(directorydata + 'testingEnsIndices_ModelBiases_%s_%s_%s_%s_iterations%s_STD-%s.txt' % (variq,monthlychoice,reg_name,dataset,iterations[0],rm_standard_dev))
classel = np.genfromtxt(directorydata + 'allClasses_ModelBiases_%s_%s_%s_%s-%s_iterations%s_STD-%s.txt' % (variq,monthlychoice,reg_name,dataset_obs,dataset,iterations[0],rm_standard_dev))

### Call data for each model
modeldata = []
for i in range(len(modelGCMs)):
    simuqq = modelGCMs[i]
    lrpq,lat1,lon1 = readData(window,typemodel,variq,simuqq,land_only,reg_name,rm_standard_dev)
    
    lrpmean = np.nanmean(lrpq,axis=0)
    modeldata.append(lrpmean)
lrpall = np.asarray(modeldata,dtype=object)

### Composite data
# diff = comp(lrpall,yearq)
