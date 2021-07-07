"""
Script to plot figure 3

Author     : Zachary M. Labe
Date       : 7 July 2021
Version    : 1 
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as c
import numpy as np
import palettable.cubehelix as cm
import palettable.scientific.sequential as sss
import palettable.cartocolors.qualitative as cc
import cmocean as cmocean
import cmasher as cmr
import calc_Utilities as UT
import scipy.stats as sts

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

### Set parameters
directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/MSFigures_v1/'
directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/MSFigures/'
variablesall = ['T2M']
allDataLabels = ['CanESM2','MPI','CSIRO-MK3.6','EC-EARTH','GFDL-CM3','GFDL-ESM2M','LENS','MM-Mean']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p"]
THRESH = 0.05

### Read in data
globe = np.load(directorydata + 'Ranks_thresh-%s_%s.npy' % (THRESH,'SMILEGlobe'))
arctic = np.load(directorydata + 'Ranks_thresh-%s_%s.npy' % (THRESH,'LowerArctic'))

###############################################################################
###############################################################################
###############################################################################
###############################################################################                      
### Plot first meshgrid
fig = plt.figure()

ax = plt.subplot(211)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.get_xaxis().set_tick_params(direction='out', width=2,length=3,
            color='darkgrey')
ax.get_yaxis().set_tick_params(direction='out', width=2,length=3,
            color='darkgrey')

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom='off')
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left='on',      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft='on')

csm=plt.get_cmap(cmocean.cm.ice)
norm = c.BoundaryNorm(np.arange(1,9+1,1),csm.N)

cs = plt.pcolormesh(globe,shading='faceted',edgecolor='darkgrey',
                    linewidth=0.05,vmin=1,vmax=8,norm=norm,cmap=csm,clip_on=False)

plt.yticks(np.arange(0.5,8.5,1),allDataLabels,ha='right',va='center',color='k',size=6)
yax = ax.get_yaxis()
yax.set_tick_params(pad=2)
plt.xticks([])
plt.xlim([0,70])

for i in range(globe.shape[0]):
    for j in range(globe.shape[1]):
        cc = 'crimson'         
        plt.text(j+0.5,i+0.5,r'\textbf{%s}' % int(globe[i,j]),fontsize=4,
            color=cc,va='center',ha='center')

###############################################################################
###############################################################################
###############################################################################
ax = plt.subplot(212)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.get_xaxis().set_tick_params(direction='out', width=2,length=3,
            color='darkgrey')
ax.get_yaxis().set_tick_params(direction='out', width=2,length=3,
            color='darkgrey')

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom='on')
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left='on',      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft='on')

csm=plt.get_cmap(cmocean.cm.ice)
norm = c.BoundaryNorm(np.arange(1,9+1,1),csm.N)

cs = plt.pcolormesh(arctic,shading='faceted',edgecolor='darkgrey',
                    linewidth=0.05,vmin=1,vmax=8,norm=norm,cmap=csm,clip_on=False)

plt.yticks(np.arange(0.5,8.5,1),allDataLabels,ha='right',va='center',color='k',size=6)
yax = ax.get_yaxis()
yax.set_tick_params(pad=2)
plt.xticks(np.arange(0.5,70.5,5),map(str,np.arange(1950,2022,5)),
            color='k',size=6)
plt.xlim([0,70])

for i in range(arctic.shape[0]):
    for j in range(arctic.shape[1]):
        cc = 'crimson'         
        plt.text(j+0.5,i+0.5,r'\textbf{%s}' % int(arctic[i,j]),fontsize=4,
            color=cc,va='center',ha='center')
        
plt.annotate(r'\textbf{GLOBAL}',
              textcoords='axes fraction',
              xy=(0,0), xytext=(1.02,1.26),
              fontsize=18,color='dimgrey',alpha=1,rotation=270,va='bottom')
plt.annotate(r'\textbf{ARCTIC}',
              textcoords='axes fraction',
              xy=(0,0), xytext=(1.02,0.17),
              fontsize=18,color='dimgrey',alpha=1,rotation=270,va='bottom')

###############################################################################                
cbar_ax1 = fig.add_axes([0.35,0.11,0.3,0.03])                
cbar = fig.colorbar(cs,cax=cbar_ax1,orientation='horizontal',
                    extend='neither',extendfrac=0.07,drawedges=True)
cbar.set_ticks([])
cbar.set_ticklabels([])  
cbar.ax.invert_xaxis()
cbar.ax.tick_params(axis='x', size=.001,labelsize=7)
cbar.dividers.set_color('dimgrey')
cbar.dividers.set_linewidth(1)
cbar.outline.set_edgecolor('dimgrey')
cbar.outline.set_linewidth(1)
cbar.set_label(r'\textbf{RELATIVE MODEL CHOICE}',color='k',labelpad=7,fontsize=18)

cbar.ax.get_yaxis().set_ticks([])
for j, lab in enumerate(range(1,9,1)):
    cbar.ax.text((2 * j+2.9)/2, 4.5, lab,ha='center',va='center',
                 size=5,color='dimgrey')

# plt.annotate(r'\textbf{aaaaaaa}',
#               textcoords='figure fraction',
#               xy=(0,0), xytext=(0.38,0.13),ha='center',va='center',
#               fontsize=5,color='crimson',alpha=1,rotation=0,zorder=100)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2,hspace=0.1)
plt.savefig(directoryfigure + 'MS-Figure_3_v1.png',dpi=1000)
        
        