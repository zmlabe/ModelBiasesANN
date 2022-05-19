"""
Script for creating composites to compare the mean state of the MMLEA

Author     : Zachary M. Labe
Date       : 25 February 2022
Version    : 7 - adds validation data for early stopping
"""

### Import packages
import sys
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import palettable.scientific.sequential as sss
import cmocean as cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import scipy.stats as sts
import matplotlib
import cmasher as cmr

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

### Parameters
directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/RevisitResults_v7/'
directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/v7/'
variablesall = 'T2M'
dataset_obs = 'ERA5BE'
scaleLRPmax = True
allDataLabels = ['CanESM2','MPI','CSIRO-MK3.6','EC-EARTH','GFDL-CM3','GFDL-ESM2M','LENS']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p"]

###############################################################################
###############################################################################
############################################################################### 
### Read in data
lat1 = np.load(directorydata + 'Lat_ArcticALL.npy',allow_pickle=True)
lon1 = np.load(directorydata + 'Lon_ArcticALL.npy',allow_pickle=True)
lat1g = np.load(directorydata + 'Lat_SMILEGlobe.npy',allow_pickle=True)
lon1g = np.load(directorydata + 'Lon_SMILEGlobe.npy',allow_pickle=True)
lrp = np.load(directorydata + 'LRPcomposites_ArcticALL_7classes_GLO_%s.npy' % dataset_obs,allow_pickle=True)

###############################################################################
###############################################################################
###############################################################################     
#######################################################################
#######################################################################
#######################################################################
### Plot subplot of mean climate
limit = np.arange(0,0.80001,0.005)
barlim = np.round(np.arange(0,0.801,0.8),2)
cmap = cm.cubehelix2_16.mpl_colormap
label = r'\textbf{RELEVANCE}'
    
fig = plt.figure(figsize=(8,2))
for r in range(0,len(allDataLabels)):
    var = lrp[r]/np.nanmax(lrp[r])
    
    ax1 = plt.subplot(1,7,r+1)
    m = Basemap(projection='npstere',boundinglat=61.5,lon_0=0,
                resolution='l',round =True,area_thresh=10000)
    m.drawcoastlines(color='darkgrey',linewidth=0.24)
        
    var, lons_cyclic = addcyclic(var, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.4)
    circle.set_clip_on(False)
    
    xx = 'max'
    cs1 = m.contourf(x,y,var,limit,extend=xx)
    cs1.set_cmap(cmap) 
            
    ax1.annotate(r'\textbf{%s}' % allDataLabels[r],xy=(0,0),xytext=(0.5,1.15),
                  textcoords='axes fraction',color='k',fontsize=9,
                  rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),
            xytext=(0.85,0.89),xycoords='axes fraction',rotation=330,
            color='dimgrey',fontsize=9)
    
###############################################################################
cbar_ax1 = fig.add_axes([0.395,0.15,0.2,0.035])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend=xx,extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='k',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.001,labelsize=7,pad=2)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.savefig(directoryfigure + 'LRPcomposites_GLOclimatemodels.png',dpi=900)
        