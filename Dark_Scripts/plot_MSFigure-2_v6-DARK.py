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

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
plt.rc('savefig',facecolor='black')
plt.rc('axes',edgecolor='darkgrey')
plt.rc('xtick',color='darkgrey')
plt.rc('ytick',color='darkgrey')
plt.rc('axes',labelcolor='darkgrey')
plt.rc('axes',facecolor='black')

### Parameters
directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/RevisitResults_v6/'
directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/v6/DARK_Figures/'
variablesall = 'T2M'
scaleLRPmax = True
allDataLabels = ['CanESM2','MPI','CSIRO-MK3.6','EC-EARTH','GFDL-CM3','GFDL-ESM2M','LENS']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p"]

###############################################################################
###############################################################################
############################################################################### 
### Read in data
lat1 = np.load(directorydata + 'Lat_SMILEGlobe.npy',allow_pickle=True)
lon1 = np.load(directorydata + 'Lon_SMILEGlobe.npy',allow_pickle=True)
rawdata = np.load(directorydata + 'MMMeandifferences_7models.npy',allow_pickle=True)
lrp = np.load(directorydata + 'LRPcomposites_SMILEglobe_7classes.npy',allow_pickle=True)

### Prepare data for plotting
alldata = np.append(lrp,rawdata,axis=0)

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of LRP means training
limit = np.arange(0,0.80001,0.005)
barlim = np.round(np.arange(0,0.801,0.8),2)
cmap = cm.cubehelix2_16.mpl_colormap
label = r'\textbf{RELEVANCE}'

limitr = np.arange(-5,5.01,0.2)
barlimr = np.round(np.arange(-5,6,5),2)
cmapr = cmocean.cm.balance
labelr = r'\textbf{$^{\circ}$C}'

fig = plt.figure(figsize=(10,3))
for r in range(alldata.shape[0]):
    if r < 7:
        var = alldata[r]
        
        if scaleLRPmax == True:
            var = var/np.nanmax(var)
        else:
            var = var
        
        ax1 = plt.subplot(2,alldata.shape[0]//2,r+1)
        m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
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
                
        ax1.annotate(r'\textbf{%s}' % allDataLabels[r],xy=(0,0),xytext=(0.5,1.25),
                      textcoords='axes fraction',color='w',fontsize=8,
                      rotation=0,ha='center',va='center')
    elif r >= 7:
        var = alldata[r]
        
        ax1 = plt.subplot(2,alldata.shape[0]//2,r+1)
        m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
        m.drawcoastlines(color='dimgrey',linewidth=0.24)
        if r == 15: 
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
    
    if r == 5:
        cbar_ax = fig.add_axes([0.91,0.55,0.011,0.14])                
        cbar = fig.colorbar(cs1,cax=cbar_ax,orientation='vertical',
                            extend='max',extendfrac=0.07,drawedges=False)    
        cbar.set_label(label,fontsize=7,color='darkgrey')      
        cbar.set_ticks(barlim)
        cbar.set_ticklabels(list(map(str,barlim)))
        cbar.ax.tick_params(labelsize=4,pad=7) 
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs,ha='center')
        cbar.ax.tick_params(axis='y', size=.001)
        cbar.outline.set_edgecolor('darkgrey')
        cbar.outline.set_linewidth(0.5)
        
    elif r == 12:
        cbar_ax = fig.add_axes([0.91,0.33,0.011,0.14])                
        cbar = fig.colorbar(cs1,cax=cbar_ax,orientation='vertical',
                            extend='both',extendfrac=0.07,drawedges=False)    
        cbar.set_label(labelr,fontsize=7,color='darkgrey',labelpad=6.5)    
        cbar.set_ticks(barlimr)
        cbar.set_ticklabels(list(map(str,barlimr)))
        cbar.ax.tick_params(labelsize=4,pad=5) 
        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs,ha='center')
        cbar.ax.tick_params(axis='y', size=.001)
        cbar.outline.set_edgecolor('darkgrey')
        cbar.outline.set_linewidth(0.5)

# plt.tight_layout()
plt.subplots_adjust(wspace=0.01,hspace=-0.61)
if scaleLRPmax == True:
    plt.savefig(directoryfigure + 'MSFigure-2_v6_scaleLRP_DARK.png',dpi=1000)
else:
    plt.savefig(directoryfigure + 'MSFigure-2_v6_DARK.png',dpi=1000)