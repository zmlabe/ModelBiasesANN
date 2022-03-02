"""
Script for creating composites to compare the scaled composites of ERA5-BE
over epochs with map mean removed

Author     : Zachary M. Labe
Date       : 2 March 2022
Version    : 7 - adds validation data for early stopping
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import scipy.stats as sts
import calc_DetrendData as DET
from netCDF4 import Dataset
import cmasher as cmr
import cmocean
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/v7/'
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
datasetsingle = ['SMILE']
dataset_obsall = ['ERA5BE']
obslabels = ['ERA5-BE']
monthlychoiceq = ['annual']
variables = ['T2M']
reg_name = 'Arctic'
level = 'surface'
timeper = 'historical'
###############################################################################
###############################################################################
land_only = False
ocean_only = False
###############################################################################
###############################################################################
baseline = np.arange(1981,2010+1,1)
###############################################################################
###############################################################################
window = 0
yearsobs = [np.arange(1950+window,2019+1,1)]
###############################################################################
###############################################################################
numOfEns = 16
###############################################################################
###############################################################################
dataset = datasetsingle[0]
lat_bounds,lon_bounds = UT.regions(reg_name)
###############################################################################
###############################################################################
ravelyearsbinary = False
ravelbinary = False
lensalso = True
randomalso = False
shuffletype = 'none'
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in model data
def read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
    data_obs,lats_obs,lons_obs = df.readFiles(variq,dataset_obs,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
    data_obs,lats_obs,lons_obs = df.getRegion(data_obs,lats_obs,lons_obs,
                                            lat_bounds,lon_bounds)    
    print('our OBS dataset: ',dataset_obs,' is shaped',data_obs.shape)
    return data_obs,lats_obs,lons_obs

###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in model data
variq = variables[0]
monthlychoice = monthlychoiceq[0]

### Only focus on 1950 to 2015/2019 for composites
meanobsq = []
for vv in range(len(dataset_obsall)):
    data_obsn,lats_obs,lons_obs = read_obs_dataset(variq,dataset_obsall[vv],numOfEns,
                                                  lensalso,randomalso,ravelyearsbinary,
                                                  ravelbinary,shuffletype,lat_bounds,
                                                  lon_bounds)
    
    meanobsq.append(data_obsn)
    
### Evaluate observations
meanobsq = np.asarray(meanobsq)

### Read lat/lon grid
lon2,lat2 = np.meshgrid(lons_obs,lats_obs)

### Remove annual mean
erac = meanobsq[0] - UT.calc_weightedAve(meanobsq[0],lat2)[:,np.newaxis,np.newaxis]
    
### Read in multimodel-mean
meanTraininge = np.genfromtxt('/Users/zlabe/Documents/Research/ModelComparison/Data/TRAININGstandardmean_historical_annual_NoiseTwinSingleMODDIF4_Experiment-4_T2M_Arctic_ERA5BE_NumOfSMILE-7_Method-ENS.txt',
                              unpack=True).reshape(lats_obs.shape[0],lons_obs.shape[0])
stdTraininge = np.genfromtxt('/Users/zlabe/Documents/Research/ModelComparison/Data/TRAININGstandardstd_historical_annual_NoiseTwinSingleMODDIF4_Experiment-4_T2M_Arctic_ERA5BE_NumOfSMILE-7_Method-ENS.txt',
                             unpack=True).reshape(lats_obs.shape[0],lons_obs.shape[0])

### Calculate composites
mmme = meanTraininge
stde = stdTraininge

### Rescale by training mean and standard deviation
z = (erac - mmme)/stde

### Calculate epochs
years = yearsobs[0]
yearq1 = np.where((years >= 1950) & (years <= 1978))[0]
yearq2 = np.where((years >= 1979) & (years <= 1999))[0]
yearq3 = np.where((years >= 2000))[0]

epoch1 = np.nanmean(z[yearq1,:,:],axis=0)
epoch2 = np.nanmean(z[yearq2,:,:],axis=0)
epoch3 = np.nanmean(z[yearq3,:,:],axis=0)

### Plotting arguments
difflimits = np.arange(-3,3.01,0.1)
diffbar = np.arange(-3,4,1)

composites = [epoch1,epoch2,epoch3]
datalabels = [r'\textbf{1950-1978}',r'\textbf{1979-1999}',r'\textbf{2000-2019}']

###########################################################################
###########################################################################
###########################################################################
### Plot variable data for DJF
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
plt.rcParams['hatch.color'] = 'k'

fig = plt.figure(figsize=(7,3.5))

for v in range(len(composites)):
    ax = plt.subplot(1,3,v+1)

    var = composites[v]
    
    m = Basemap(projection='npstere',boundinglat=61.5,lon_0=0,
                            resolution='l',round =True,area_thresh=10000)

    var, lons_cyclic = addcyclic(var, lons_obs)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lats_obs)
    x, y = m(lon2d, lat2d)
              
    circle = m.drawmapboundary(fill_color='white',color='dimgrey',linewidth=0.7)
    circle.set_clip_on(False)

    cs = m.contourf(x,y,var,difflimits,extend='both')

    m.drawcoastlines(color='dimgrey',linewidth=0.8)
            
    ### Add experiment text to subplot
    if any([v == 0,v == 1,v == 2]):
        ax.annotate(r'\textbf{%s}' % datalabels[v],xy=(0,0),xytext=(0.5,1.10),
                      textcoords='axes fraction',color='dimgrey',
                      fontsize=15,rotation=0,ha='center',va='center')
    
    cs.set_cmap(cmocean.cm.balance) 
        
    ax.annotate(r'\textbf{[%s]}' % letters[v],xy=(0,0),
            xytext=(0.88,0.9),xycoords='axes fraction',
            color='k',fontsize=7)
        
    ax.set_aspect('equal')
            

    cbar_ax = fig.add_axes([0.35,0.14,0.3,0.015])                 
    cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                        extend='both',extendfrac=0.07,drawedges=False)    
    cbar.set_label(r'\textbf{T2M-Scaled-RM}',fontsize=10,color='k')   
    cbar.set_ticks(diffbar)
    cbar.set_ticklabels(list(map(str,diffbar)))
    cbar.ax.tick_params(labelsize=6,pad=3) 
    ticklabs = cbar.ax.get_xticklabels()
    cbar.ax.set_xticklabels(ticklabs,ha='center')
    cbar.ax.tick_params(axis='x', size=.001)
    cbar.outline.set_edgecolor('dimgrey')
    cbar.outline.set_linewidth(0.5)

plt.tight_layout()    
fig.subplots_adjust(bottom=0.11,hspace=0.01,wspace=0.01)
    
plt.savefig(directoryfigure + 'ERA5BE_ScaledEpochs-RM.png',dpi=900)

print('Completed: Script done!')