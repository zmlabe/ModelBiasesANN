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
modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth',
              'GFDL-CM3','GFDL-ESM2M','LENS']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
datasetsingle = ['SMILE']
dataset = datasetsingle[0]
dataset_obs = '20CRv3'
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
    
    data = Dataset(directorydata + 'LRP_Maps_ModelBiases-STDDEV%syrs_%s_Annual_%s_%s_land_only-%s_%s_STD-%s.nc' % (window,typemodel,variq,simuqq,land_only,reg_name,rm_standard_dev))
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    lrp = data.variables['LRP'][:]
    data.close()
    
    return lrp,lat,lon

### Read in prediction data
trainindices = np.int_(np.genfromtxt(directorydata + 'trainingEnsIndices_ModelBiases_%s_%s_%s_%s_iterations%s_STD-%s.txt' % (variq,monthlychoice,reg_name,dataset,iterations[0],rm_standard_dev)))
testindices = np.int_(np.genfromtxt(directorydata + 'testingEnsIndices_ModelBiases_%s_%s_%s_%s_iterations%s_STD-%s.txt' % (variq,monthlychoice,reg_name,dataset,iterations[0],rm_standard_dev)))
classel = np.int_(np.genfromtxt(directorydata + 'allClasses_ModelBiases_%s_%s_%s_%s-%s_iterations%s_STD-%s.txt' % (variq,monthlychoice,reg_name,dataset_obs,dataset,iterations[0],rm_standard_dev)))

### Create lists for loops
xindices = [trainindices,testindices]

### Call data for each model
modeldata = []
typemodel = ['train','test']
for i in range(len(typemodel)):
    lrpq,lat1,lon1 = readData(window,typemodel[i],variq,simuqq,land_only,reg_name,rm_standard_dev)
    modeldata.append(lrpq)
lrpall = np.asarray(modeldata,dtype=object)

### Find which model
model_train = []
for i in range(len(modelGCMs)):
    modelloc = np.where((classel[xindices[0]] == int(i)))[0]
    model_train.append(modelloc)
    
model_test = []
for i in range(len(modelGCMs)):
    modelloc = np.where((classel[xindices[1]] == int(i)))[0]
    model_test.append(modelloc)
    
### Composite lrp maps of the models
lrptrain = []
for i in range(len(modelGCMs)):
    lrpmodel = lrpall[0][model_train[i]]
    lrpmodelmean = np.nanmean(lrpmodel,axis=0)
    lrptrain.append(lrpmodelmean)
lrptraina = np.asarray(lrptrain,dtype=object)

lrptest = []
for i in range(len(modelGCMs)):
    lrpmodel = lrpall[1][model_test[i]]
    lrpmodelmean = np.nanmean(lrpmodel,axis=0)
    lrptest.append(lrpmodelmean)
lrptesta = np.asarray(lrptest,dtype=object)

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of LRP means training
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

limit = np.arange(0,0.4001,0.005)
barlim = np.round(np.arange(0,0.5,0.1),2)
cmap = cm.cubehelix2_16.mpl_colormap
label = r'\textbf{RELEVANCE}'

fig = plt.figure(figsize=(10,2))
for r in range(len(modelGCMs)):
    var = lrptraina[r]
    
    ax1 = plt.subplot(1,7,r+1)
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    m.drawcoastlines(color='darkgrey',linewidth=0.27)
        
    var, lons_cyclic = addcyclic(var, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    cs1 = m.contourf(x,y,var,limit,extend='max')
    cs1.set_cmap(cmap) 
            
    ax1.annotate(r'\textbf{%s}' % modelGCMs[r],xy=(0,0),xytext=(0.5,1.10),
                  textcoords='axes fraction',color='dimgrey',fontsize=8,
                  rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                  textcoords='axes fraction',color='k',fontsize=6,
                  rotation=330,ha='center',va='center')
    
###############################################################################
cbar_ax1 = fig.add_axes([0.36,0.15,0.3,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)

plt.savefig(directoryfigure + 'LRP_Composites_ModelBiases_Training.png',dpi=300)

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of LRP means testing
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

limit = np.arange(0,0.4001,0.005)
barlim = np.round(np.arange(0,0.5,0.1),2)
cmap = cm.cubehelix2_16.mpl_colormap
label = r'\textbf{RELEVANCE}'

fig = plt.figure(figsize=(10,2))
for r in range(len(modelGCMs)):
    var = lrptesta[r]
    
    ax1 = plt.subplot(1,7,r+1)
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    m.drawcoastlines(color='darkgrey',linewidth=0.27)
        
    var, lons_cyclic = addcyclic(var, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    cs1 = m.contourf(x,y,var,limit,extend='max')
    cs1.set_cmap(cmap) 
            
    ax1.annotate(r'\textbf{%s}' % modelGCMs[r],xy=(0,0),xytext=(0.5,1.10),
                  textcoords='axes fraction',color='dimgrey',fontsize=8,
                  rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                  textcoords='axes fraction',color='k',fontsize=6,
                  rotation=330,ha='center',va='center')
    
###############################################################################
cbar_ax1 = fig.add_axes([0.36,0.15,0.3,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)

plt.savefig(directoryfigure + 'LRP_Composites_ModelBiases_Testing.png',dpi=300)

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of LRP means both training and testing
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

limit = np.arange(0,0.4001,0.005)
barlim = np.round(np.arange(0,0.5,0.1),2)
cmap = cm.cubehelix2_16.mpl_colormap
label = r'\textbf{RELEVANCE}'

fig = plt.figure(figsize=(10,2))
for r in range(len(modelGCMs)):
    var = (lrptraina[r]+lrptesta[r])/2.
    
    ax1 = plt.subplot(1,7,r+1)
    m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
    m.drawcoastlines(color='darkgrey',linewidth=0.27)
        
    var, lons_cyclic = addcyclic(var, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    cs1 = m.contourf(x,y,var,limit,extend='max')
    cs1.set_cmap(cmap) 
            
    ax1.annotate(r'\textbf{%s}' % modelGCMs[r],xy=(0,0),xytext=(0.5,1.10),
                  textcoords='axes fraction',color='dimgrey',fontsize=8,
                  rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                  textcoords='axes fraction',color='k',fontsize=6,
                  rotation=330,ha='center',va='center')
    
###############################################################################
cbar_ax1 = fig.add_axes([0.36,0.15,0.3,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(top=0.85,wspace=0.02,hspace=0.02,bottom=0.14)

plt.savefig(directoryfigure + 'LRP_Composites_ModelBiases_Combined.png',dpi=300)

#######################################################################
#######################################################################
#######################################################################
### Plot subplot of LRP for ALL DATA
limit = np.arange(0,0.20001,0.0005)
barlim = np.round(np.arange(0,0.3,0.1),2)
cmap = cm.cubehelix2_16.mpl_colormap
label = r'\textbf{RELEVANCE}'

### Calculate total composite
lrptotalcomp = np.append(lrpall[0],lrpall[1],axis=0)
lrptotalcompmean = np.nanmean(lrptotalcomp,axis=0)

fig = plt.figure()
var = lrptotalcompmean

ax1 = plt.subplot(111)
m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.4)
    
var, lons_cyclic = addcyclic(var, lon1)
var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
x, y = m(lon2d, lat2d)
   
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,var,limit,extend='max')
cs1.set_cmap(cmap) 

ax1.annotate(r'\textbf{TOTAL LRP}',xy=(0,0),xytext=(0.82,0.94),
              textcoords='axes fraction',color='k',fontsize=12,
              rotation=340,ha='center',va='center')
    
###############################################################################
cbar_ax1 = fig.add_axes([0.36,0.15,0.3,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(bottom=0.14)

plt.savefig(directoryfigure + 'LRP_ALLComposites_ModelBiases_Combined',dpi=300)