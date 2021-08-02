"""
Script to plot figure 4b

Author     : Zachary M. Labe
Date       : 12 July 2021
Version    : 2
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
directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/MSFigures_v2/'
directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/MSFigures_v2/'
variablesall = ['T2M']
yearsall = np.arange(1950,2019+1,1)
allDataLabels = ['CanESM2','MPI','CSIRO-MK3.6','EC-EARTH','GFDL-CM3','GFDL-ESM2M','LENS','MM-Mean']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p"]

### Read in frequency data
globef = np.load(directorydata + 'CountingIterations_%s.npz' % ('SMILEGlobe'))
arcticf = np.load(directorydata + 'CountingIterations_%s.npz' % ('LowerArctic'))

gmeanff = globef['mmean']
ggfdlff = globef['gfdlcm']

ameanff = arcticf['mmean']
agfdlff = arcticf['gfdlcm']

###############################################################################
###############################################################################
###############################################################################
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')  
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
            ax.xaxis.set_ticks([]) 

### Begin plot
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(211)

adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.35,clip_on=False,linewidth=0.5)

x=np.arange(1950,2019+1,1)
plt.plot(yearsall,gmeanff,linewidth=5,color='k',alpha=1,zorder=3,clip_on=False)

plt.yticks(np.arange(0,101,10),map(str,np.round(np.arange(0,101,10),2)),size=9)
plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=9)
plt.xlim([1950,2020])   
plt.ylim([0,100])  

plt.text(1949,104,r'\textbf{[a]}',color='dimgrey',
         fontsize=7,ha='center')  
plt.text(2022,50,r'\textbf{GLOBAL}',color='dimgrey',fontsize=25,rotation=270,
         ha='center',va='center')
plt.ylabel(r'\textbf{Frequency of Label}',color='k',fontsize=10)         

###############################################################################
ax = plt.subplot(212)

adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.35,clip_on=False,linewidth=0.5)

x=np.arange(1950,2019+1,1)
plt.plot(yearsall,ameanff,linewidth=5,color='k',alpha=1,zorder=3,clip_on=False,label=r'\textbf{MM-Mean}')
plt.plot(yearsall,agfdlff,linewidth=4,color=plt.cm.CMRmap(0.6),alpha=1,zorder=3,clip_on=False,label=r'\textbf{GFDL-CM3}',
         linestyle='--',dashes=(1,0.3))

plt.yticks(np.arange(0,101,10),map(str,np.round(np.arange(0,101,10),2)),size=9)
plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=9)
plt.xlim([1950,2020])   
plt.ylim([0,100])  

plt.text(1949,104,r'\textbf{[b]}',color='dimgrey',
         fontsize=7,ha='center')  
leg = plt.legend(shadow=False,fontsize=11,loc='upper center',
              bbox_to_anchor=(0.5,1.22),fancybox=True,ncol=4,frameon=False,
              handlelength=5,handletextpad=1)
plt.ylabel(r'\textbf{Frequency of Label}',color='k',fontsize=10) 
plt.text(2022,50,r'\textbf{ARCTIC}',color='dimgrey',fontsize=25,rotation=270,
         ha='center',va='center')          
        
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.savefig(directoryfigure + 'MS-Figure_4b_v2_Poster.png',dpi=1000)