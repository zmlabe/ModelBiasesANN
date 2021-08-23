"""
Script to plot figure 5

Author     : Zachary M. Labe
Date       : 7 July 2021
Version    : 1 
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

### Parameters
directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/MSFigures_v2/'
directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/MSFigures_v2/'
variablesall = 'T2M'
scaleLRPmax = True
allDataLabels = ['CanESM2','MPI','CSIRO-MK3.6','EC-EARTH','GFDL-CM3','GFDL-ESM2M','LENS','MM-Mean']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p"]

###############################################################################
###############################################################################
############################################################################### 
### Read in data
lat1 = np.load(directorydata + 'Lat_LowerArctic.npy',allow_pickle=True)
lon1 = np.load(directorydata + 'Lon_LowerArctic.npy',allow_pickle=True)
lrp = np.load(directorydata + 'LRPcomposites_LowerArcticAA_8classes.npy',allow_pickle=True)

### Prepare data for plotting
alldata = lrp

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of LRP means training
limit = np.arange(0,0.60001,0.005)
barlim = np.round(np.arange(0,0.601,0.6),2)
cmap = cm.cubehelix2_16.mpl_colormap
label = r'\textbf{RELEVANCE}'

fig = plt.figure(figsize=(8,8))
for r in range(alldata.shape[0]):
    var = alldata[r]
    
    if scaleLRPmax == True:
        var = var/np.nanmax(var)
    else:
        var = var
    
    ax1 = plt.subplot(2,alldata.shape[0]-4,r+1)
    m = Basemap(projection='npstere',boundinglat=61.3,lon_0=0,
                resolution='l',round =True,area_thresh=10000)
    m.drawcoastlines(color='darkgrey',linewidth=0.45)
        
    var, lons_cyclic = addcyclic(var, lon1)
    var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
    lon2d, lat2d = np.meshgrid(lons_cyclic, lat1)
    x, y = m(lon2d, lat2d)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    cs1 = m.contourf(x,y,var,limit,extend='max')
    cs1.set_cmap(cmap) 
            
    ax1.annotate(r'\textbf{%s}' % allDataLabels[r],xy=(0,0),xytext=(0.5,1.04),
                  textcoords='axes fraction',color='dimgrey',fontsize=13,
                  rotation=0,ha='center',va='center')
    # ax1.annotate(r'\textbf{[%s]}' % letters[r],xy=(0,0),xytext=(0.86,0.97),
    #               textcoords='axes fraction',color='k',fontsize=6,
    #               rotation=330,ha='center',va='center')
    
###############################################################################
cbar_ax1 = fig.add_axes([0.36,0.19,0.3,0.03])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=9,color='dimgrey',labelpad=0.7)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=5)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(hspace=-0.6,wspace=0.02)

if scaleLRPmax == True:
    plt.savefig(directoryfigure + 'MS-Figure_5a_v2_scaleLRP_POSTER.png',dpi=1000)
else:
    plt.savefig(directoryfigure + 'MS-Figure_5a_v2_POSTER.png',dpi=1000)