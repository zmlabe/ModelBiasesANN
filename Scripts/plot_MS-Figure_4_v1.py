"""
Script to plot figure 4

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
yearsall = np.arange(1950,2019+1,1)
allDataLabels = ['CanESM2','MPI','CSIRO-MK3.6','EC-EARTH','GFDL-CM3','GFDL-ESM2M','LENS','MM-Mean']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p"]

### Read in confidence data
globe = np.load(directorydata + 'StatisticsIterations_%s.npz' % ('SMILEGlobe'))
arctic = np.load(directorydata + 'StatisticsIterations_%s.npz' % ('LowerArctic'))

gmean = globe['mean']
g5 = globe['p5']
g95 = globe['p95']

amean = arctic['mean']
a5 = arctic['p5']
a95 = arctic['p95']

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
fig = plt.figure(figsize=(8,5))
ax = plt.subplot(221)

mean = gmean
lower = g5
upper = g95

adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.35,clip_on=False,linewidth=0.5)

plt.plot(yearsall,mean,linewidth=2,color='crimson',alpha=1,zorder=3,clip_on=False)
ax.fill_between(yearsall,lower,mean,facecolor='crimson',alpha=0.35,zorder=1,clip_on=False)
ax.fill_between(yearsall,mean,upper,facecolor='crimson',alpha=0.35,zorder=1,clip_on=False)

plt.yticks(np.arange(0,1.01,0.1),map(str,np.round(np.arange(0,1.01,0.1),2)),size=5)
plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5)
plt.xlim([1950,2020])   
plt.ylim([0.4,1.0])  

plt.text(1950,0.41,r'\textbf{GLOBAL}',color='dimgrey',
         fontsize=22,ha='left')  
plt.text(1948,1.03,r'\textbf{[a]}',color='dimgrey',
         fontsize=7,ha='center')  
plt.ylabel(r'\textbf{Maximum Confidence}',color='k',fontsize=7)   

###############################################################################
ax = plt.subplot(222)

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
plt.plot(yearsall,gmeanff,linewidth=2,color='k',alpha=1,zorder=3,clip_on=False)

plt.yticks(np.arange(0,101,10),map(str,np.round(np.arange(0,101,10),2)),size=5)
plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5)
plt.xlim([1950,2020])   
plt.ylim([0,100])  

plt.text(1948,104,r'\textbf{[b]}',color='dimgrey',
         fontsize=7,ha='center')  
plt.ylabel(r'\textbf{Frequency of Label}',color='k',fontsize=7)         

###############################################################################
ax = plt.subplot(223)

mean = amean
lower = a5
upper = a95

adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.35,clip_on=False,linewidth=0.5)

plt.plot(yearsall,mean,linewidth=2,color='deepskyblue',alpha=1,zorder=3,clip_on=False)
ax.fill_between(yearsall,lower,mean,facecolor='deepskyblue',alpha=0.35,zorder=1,clip_on=False)
ax.fill_between(yearsall,mean,upper,facecolor='deepskyblue',alpha=0.35,zorder=1,clip_on=False)

plt.yticks(np.arange(0,1.01,0.1),map(str,np.round(np.arange(0,1.01,0.1),2)),size=5)
plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5)
plt.xlim([1950,2020])   
plt.ylim([0.4,1.0])  

plt.text(1950,0.41,r'\textbf{ARCTIC}',color='dimgrey',
         fontsize=22,ha='left')  
plt.text(1948,1.03,r'\textbf{[c]}',color='dimgrey',
         fontsize=7,ha='center')  
plt.ylabel(r'\textbf{Maximum Confidence}',color='k',fontsize=7)

###############################################################################
ax = plt.subplot(224)

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
plt.plot(yearsall,ameanff,linewidth=2,color='k',alpha=1,zorder=3,clip_on=False,label=r'\textbf{MM-Mean}')
plt.plot(yearsall,agfdlff,linewidth=1.5,color='maroon',alpha=1,zorder=3,clip_on=False,label=r'\textbf{GFDL-CM3}',
         linestyle='--',dashes=(1,0.3))

plt.yticks(np.arange(0,101,10),map(str,np.round(np.arange(0,101,10),2)),size=5)
plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5)
plt.xlim([1950,2020])   
plt.ylim([0,100])  

plt.text(1948,104,r'\textbf{[d]}',color='dimgrey',
         fontsize=7,ha='center')  
leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
              bbox_to_anchor=(0.5,1.2),fancybox=True,ncol=4,frameon=False,
              handlelength=5,handletextpad=1)
plt.ylabel(r'\textbf{Frequency of Label}',color='k',fontsize=7)               
        
plt.tight_layout()
plt.savefig(directoryfigure + 'MS-Figure_4_v1.png',dpi=1000)