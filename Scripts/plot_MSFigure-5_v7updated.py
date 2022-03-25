"""
Script to plot figure 2

Author     : Zachary M. Labe
Date       : 2 December 2021
Version    : 6 - standardizes observations by training data (only 7 models)
"""

### Import packages
import matplotlib.pyplot as plt
import numpy as np
import sys
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import palettable.cartocolors.qualitative as cc
from sklearn.metrics import accuracy_score
import scipy.stats as sts
import cmocean
import itertools

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
lrp = np.load(directorydata + 'LRPobs_ArcticALL_7classes_%s.npy' % dataset_obs,allow_pickle=True)
rawdata = np.load(directorydata + 'LRPobs_OBSCALED_ArcticALL_7classes_%s.npy' % dataset_obs,allow_pickle=True)
obs_test = np.load(directorydata + 'Labels_LRPObs_%s.npy' % dataset_obs,allow_pickle=True)

### Prepare data for plotting
alldata = list(itertools.chain(*[lrp,rawdata]))

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of LRP means training
limit = np.arange(0,0.30001,0.005)
barlim = np.round(np.arange(0,0.301,0.3),2)
cmap = cm.cubehelix2_16.mpl_colormap
label = r'\textbf{RELEVANCE}'

limitr = np.arange(-2,2.01,0.01)
barlimr = np.round(np.arange(-2,3,2),2)
cmapr = cmocean.cm.balance
labelr = r'\textbf{T2M-Scaled [$^{\circ}$C]}'

fig = plt.figure(figsize=(10,3))
for r in range(lrp.shape[0]*2):
    if r < 7:
        var = alldata[r]
        
        if (scaleLRPmax == True) & (len(obs_test[r]) > 1):
            var = var/np.nanmax(var)
        else:
            var = var
        
        ax1 = plt.subplot(2,lrp.shape[0],r+1)
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
        
        cs1 = m.contourf(x,y,var,limit,extend='max')
        cs1.set_cmap(cmap) 
                
        ax1.annotate(r'\textbf{%s}' % allDataLabels[r],xy=(0,0),xytext=(0.5,1.1),
                      textcoords='axes fraction',color='dimgrey',fontsize=8,
                      rotation=0,ha='center',va='center')
        ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                      textcoords='axes fraction',color='k',fontsize=9,
                      rotation=330,ha='center',va='center')
        ax1.annotate(r'\textbf{[%s]}' % len(obs_test[r]),xy=(0,0),xytext=(0.09,0.97),
                      textcoords='axes fraction',color=cmap(0.4),fontsize=6,
                      rotation=0,ha='center',va='center')
            
    elif r >= 7:
        var = alldata[r]
        
        ax1 = plt.subplot(2,rawdata.shape[0],r+1)
        m = Basemap(projection='npstere',boundinglat=61.5,lon_0=0,
                    resolution='l',round =True,area_thresh=10000)
        m.drawcoastlines(color='darkgrey',linewidth=0.24,zorder=20)
            
        var, lons_cyclic = addcyclic(var, lon1)
        var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
        lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
        x, y = m(lon2d, lat2d)
           
        circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                          linewidth=0.4)
        circle.set_clip_on(False)
        
        cs1 = m.contourf(x,y,var,limitr,extend='both')
        cs1.set_cmap(cmapr) 
                
        m.drawcoastlines(color='dimgrey',linewidth=0.27)
        ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
                      textcoords='axes fraction',color='k',fontsize=9,
                      rotation=330,ha='center',va='center')
        ax1.annotate(r'\textbf{[%s]}' % len(obs_test[r-7]),xy=(0,0),xytext=(0.09,0.97),
                      textcoords='axes fraction',color=cmapr(0.4),fontsize=6,
                      rotation=0,ha='center',va='center')
    
    if r == 5:
        cbar_ax = fig.add_axes([0.91,0.63,0.011,0.14])                
        cbar = fig.colorbar(cs1,cax=cbar_ax,orientation='vertical',
                            extend='max',extendfrac=0.07,drawedges=False)    
        cbar.set_label(label,fontsize=7,color='k')      
        cbar.set_ticks(barlim)
        cbar.set_ticklabels(list(map(str,barlim)))
        cbar.ax.tick_params(labelsize=4,pad=7) 
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs,ha='center')
        cbar.ax.tick_params(axis='y', size=.001)
        cbar.outline.set_edgecolor('dimgrey')
        cbar.outline.set_linewidth(0.5)
        
    elif r == 12:
        cbar_ax = fig.add_axes([0.91,0.25,0.011,0.14])                
        cbar = fig.colorbar(cs1,cax=cbar_ax,orientation='vertical',
                            extend='both',extendfrac=0.07,drawedges=False)    
        cbar.set_label(labelr,fontsize=7,color='k',labelpad=6.5)    
        cbar.set_ticks(barlimr)
        cbar.set_ticklabels(list(map(str,barlimr)))
        cbar.ax.tick_params(labelsize=4,pad=5) 
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs,ha='center')
        cbar.ax.tick_params(axis='y', size=.001)
        cbar.outline.set_edgecolor('dimgrey')
        cbar.outline.set_linewidth(0.5)

# plt.tight_layout()
plt.subplots_adjust(wspace=0.01,hspace=0)
if scaleLRPmax == True:
    plt.savefig(directoryfigure + 'MSFigure-5updated_v7_scaleLRP.png',dpi=1000)
else:
    plt.savefig(directoryfigure + 'MSFigure-5updated_v7.png',dpi=1000)