"""
Script for creating composites to compare the mean state of ERA5-BE and 20CRv3 
relative to the training mean

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
dataset_obsall = ['ERA5BE','20CRv3']
obslabels = ['ERA5-BE','20CRv3']
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
yearsobs = [np.arange(1950+window,2019+1,1),np.arange(1950+window,2015+1,1)]
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
    
    ### Calculate composites over all years
    data_obsmean = np.nanmean(data_obsn,axis=0)
    
    meanobsq.append(data_obsmean)
    
### Evaluate observations
meanobsq = np.asarray(meanobsq)
    
### Read in multimodel-mean
meanTraininge = np.genfromtxt('/Users/zlabe/Documents/Research/ModelComparison/Data/TRAININGstandardmean_historical_annual_NoiseTwinSingleMODDIF4_Experiment-3_T2M_Arctic_ERA5BE_NumOfSMILE-7_Method-ENS.txt',
                                          unpack=True).reshape(lats_obs.shape[0],lons_obs.shape[0])
meanTrainingc = np.genfromtxt('/Users/zlabe/Documents/Research/ModelComparison/Data/TRAININGstandardmean_historical_annual_NoiseTwinSingleMODDIF4_Experiment-3_T2M_Arctic_20CRv3_NumOfSMILE-7_Method-ENS.txt',
                                          unpack=True).reshape(lats_obs.shape[0],lons_obs.shape[0])

### Calculate composites
erac = meanobsq[0]
c20c = meanobsq[1]
mmme = meanTraininge
mmmc = meanTrainingc

erad = erac - mmme
c20d = c20c - mmmc
    
### Read lat/lon grid
lon2,lat2 = np.meshgrid(lons_obs,lats_obs)

### Plotting arguments
difflimits = np.arange(-10,10.01,0.1)
diffbar = np.arange(-10,11,5)

complimits = np.arange(-20,10.01,1)
compbar = np.arange(-20,11,10)

composites = [erac,mmme,erad,
              c20c,mmmc,c20d]
colormapsq = [plt.cm.twilight,plt.cm.twilight,cmocean.cm.balance,
              plt.cm.twilight,plt.cm.twilight,cmocean.cm.balance]
limitq = [complimits,complimits,difflimits,
          complimits,complimits,difflimits]
barq = [compbar,compbar,diffbar,
        compbar,compbar,diffbar]
datalabels = [r'\textbf{Observations}',r'\textbf{Training Mean}',r'\textbf{Difference']
labelstime = [r'\textbf{ERA5-BE}',r'\textbf{ERA5-BE}',r'\textbf{ERA5-BE}',
              r'\textbf{20CRv3}',r'\textbf{20CRv3}',r'\textbf{20CRv3}']

###########################################################################
###########################################################################
###########################################################################
### Plot variable data for DJF
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
plt.rcParams['hatch.color'] = 'k'

fig = plt.figure()

for v in range(6):
    ax = plt.subplot(2,3,v+1)

    var = composites[v]
    limit = limitq[v]
    
    m = Basemap(projection='npstere',boundinglat=61.5,lon_0=0,
                            resolution='l',round =True,area_thresh=10000)

    var, lons_cyclic = addcyclic(var, lons_obs)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lats_obs)
    x, y = m(lon2d, lat2d)
              
    circle = m.drawmapboundary(fill_color='white',color='dimgrey',linewidth=0.7)
    circle.set_clip_on(False)

    cs = m.contourf(x,y,var,limit,extend='both')

    m.drawcoastlines(color='dimgrey',linewidth=0.6)
            
    ### Add experiment text to subplot
    if any([v == 0,v == 3]):
        ax.annotate(r'\textbf{%s}' % labelstime[v],xy=(0,0),xytext=(-0.1,0.5),
                      textcoords='axes fraction',color='k',
                      fontsize=9,rotation=90,ha='center',va='center')
    if any([v == 0,v == 1,v == 2]):
        ax.annotate(r'\textbf{%s}' % datalabels[v],xy=(0,0),xytext=(0.5,1.12),
                      textcoords='axes fraction',color='dimgrey',
                      fontsize=15,rotation=0,ha='center',va='center')
    
    cs.set_cmap(colormapsq[v]) 
        
    ax.annotate(r'\textbf{[%s]}' % letters[v],xy=(0,0),
            xytext=(0.92,0.9),xycoords='axes fraction',
            color='k',fontsize=7)
        
    ax.set_aspect('equal')
            
    ###########################################################################
    if v == 3:
        cbar_ax = fig.add_axes([0.27,0.08,0.2,0.015])                
        cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                            extend='both',extendfrac=0.07,drawedges=False)    

        cbar.set_label(r'\textbf{T2M [$^{\circ}$C]}',fontsize=9,color='k')   
        cbar.set_ticks(barq[v])
        cbar.set_ticklabels(list(map(str,barq[v])))
        cbar.ax.tick_params(labelsize=6,pad=3) 
        ticklabs = cbar.ax.get_xticklabels()
        cbar.ax.set_xticklabels(ticklabs,ha='center')
        cbar.ax.tick_params(axis='x', size=.001)
        cbar.outline.set_edgecolor('dimgrey')
        cbar.outline.set_linewidth(0.5)
        
    elif v == 5:
        cbar_ax = fig.add_axes([0.725,0.08,0.2,0.015])                 
        cbar = fig.colorbar(cs,cax=cbar_ax,orientation='horizontal',
                            extend='both',extendfrac=0.07,drawedges=False)    
        cbar.set_label(r'\textbf{T2M Difference [$^{\circ}$C]}',fontsize=9,color='k')   
        cbar.set_ticks(barq[v])
        cbar.set_ticklabels(list(map(str,barq[v])))
        cbar.ax.tick_params(labelsize=6,pad=3) 
        ticklabs = cbar.ax.get_xticklabels()
        cbar.ax.set_xticklabels(ticklabs,ha='center')
        cbar.ax.tick_params(axis='x', size=.001)
        cbar.outline.set_edgecolor('dimgrey')
        cbar.outline.set_linewidth(0.5)

plt.tight_layout()    
fig.subplots_adjust(bottom=0.11,hspace=0.01,wspace=0.00)
    
plt.savefig(directoryfigure + 'ComparingReanalysis_mmeantraining.png',dpi=900)

print('Completed: Script done!')