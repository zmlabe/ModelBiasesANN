"""
Script to plot figure 4a

Author     : Zachary M. Labe
Date       : 16 December 2021
Version    : 7 - adds validation data for early stopping
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
directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/RevisitResults_v7/'
directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/v7/'
variablesall = ['T2M']
yearsall = np.arange(1950,2019+1,1)
allDataLabels = ['CanESM2','MPI','CSIRO-MK3.6','EC-EARTH','GFDL-CM3','GFDL-ESM2M','LENS']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p"]

### Read in confidence data
conf_globe = np.load(directorydata + 'Confidence_%s.npy' % 'Arctic')
label_globe = np.load(directorydata + 'Label_%s.npy' % 'Arctic')
conf_arctic = np.load(directorydata + 'Confidence_%s_GLO.npy' % 'Arctic')
label_arctic = np.load(directorydata + 'Label_%s_GLO.npy' % 'Arctic')

### Read in frequency data
globef = np.load(directorydata + 'CountingIterations_%s.npz' % ('Arctic'))
arcticf = np.load(directorydata + 'CountingIterations_%s_GLO.npz' % ('Arctic'))

gmpi = globef['mpi'] 
glens = globef['gfdl']

ampi = arcticf['mpi']
agfdl = arcticf['gfdl']

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
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
# ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.3)

color = cmr.infinity(np.linspace(0.00,1,len(allDataLabels)))
for i,c in zip(range(len(allDataLabels)),color):
    if i == 6:
        c = 'k'
    elif allDataLabels[i] == 'MPI':
        colormpi = c
    elif allDataLabels[i] == 'LENS':
        colorlens = 'k'
    elif allDataLabels[i] == 'GFDL-CM3':
        colorgfdl = c
    else:
        c = c
    plt.plot(yearsall,conf_globe[:,i],color=c,linewidth=0.3,
                label=r'\textbf{%s}' % allDataLabels[i],zorder=11,
                clip_on=False,alpha=1)
    plt.scatter(yearsall,conf_globe[:,i],color=c,s=28,zorder=12,
                clip_on=False,alpha=0.2,edgecolors='none')
    
    for yr in range(yearsall.shape[0]):
        la = label_globe[yr]
        if i == la:
            plt.scatter(yearsall[yr],conf_globe[yr,i],color=c,s=28,zorder=12,
                        clip_on=False,alpha=1,edgecolors='none')
        
leg = plt.legend(shadow=False,fontsize=6,loc='upper center',
              bbox_to_anchor=(0.5,-0.17),fancybox=True,ncol=4,frameon=False,
              handlelength=0,handletextpad=0)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5)
plt.yticks(np.arange(0,1.01,0.1),map(str,np.round(np.arange(0,1.01,0.1),2)),size=5)
plt.xlim([1950,2020])   
plt.ylim([0,1.0])           

plt.text(1948,1.03,r'\textbf{[a]}',color='dimgrey',
         fontsize=7,ha='center')  
plt.ylabel(r'\textbf{Confidence}',color='k',fontsize=7)

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
plt.plot(yearsall,gmpi,linewidth=2,color=colormpi,alpha=1,zorder=3,clip_on=False,label=r'\textbf{MPI}')
plt.plot(yearsall,glens,linewidth=1.5,color=colorgfdl,alpha=1,zorder=3,clip_on=False,label=r'\textbf{GFDL-CM3}',
         linestyle='--',dashes=(1,0.3))

plt.yticks(np.arange(0,101,10),map(str,np.round(np.arange(0,101,10),2)),size=5)
plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5)
plt.xlim([1950,2020])   
plt.ylim([0,100])  

# leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
#               bbox_to_anchor=(0.5,1.23),fancybox=True,ncol=4,frameon=False,
#               handlelength=5,handletextpad=1)

plt.text(1948,104,r'\textbf{[b]}',color='dimgrey',
         fontsize=7,ha='center')  
plt.text(2024,50,r'\textbf{ARCTIC}',color='dimgrey',fontsize=15,rotation=270,
         ha='center',va='center')
plt.ylabel(r'\textbf{Frequency of Label}',color='k',fontsize=7)         

###############################################################################
ax = plt.subplot(223)

adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
# ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.3)

color = cmr.infinity(np.linspace(0.00,1,len(allDataLabels)))
for i,c in zip(range(len(allDataLabels)),color):
    if i == 6:
        c = 'k'
    elif allDataLabels[i] == 'MPI':
        colormpi = c
    elif allDataLabels[i] == 'LENS':
        colorlens = 'k'
    elif allDataLabels[i] == 'GFDL-CM3':
        colorgfdl = c
    else:
        c = c
    plt.plot(yearsall,conf_arctic[:,i],color=c,linewidth=0.3,
                label=r'\textbf{%s}' % allDataLabels[i],zorder=11,
                clip_on=False,alpha=1)
    plt.scatter(yearsall,conf_arctic[:,i],color=c,s=28,zorder=12,
                clip_on=False,alpha=0.2,edgecolors='none')
    
    for yr in range(yearsall.shape[0]):
        la = label_arctic[yr]
        if i == la:
            plt.scatter(yearsall[yr],conf_arctic[yr,i],color=c,s=28,zorder=12,
                        clip_on=False,alpha=1,edgecolors='none')

plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5)
plt.yticks(np.arange(0,1.01,0.1),map(str,np.round(np.arange(0,1.01,0.1),2)),size=5)
plt.xlim([1950,2020])   
plt.ylim([0,1.0])           

plt.text(1948,1.03,r'\textbf{[c]}',color='dimgrey',
         fontsize=7,ha='center')  
plt.ylabel(r'\textbf{Confidence}',color='k',fontsize=7)   

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
plt.plot(yearsall,ampi,linewidth=2,color=colormpi,alpha=1,zorder=3,clip_on=False,label=r'\textbf{MPI}')
plt.plot(yearsall,agfdl,linewidth=1.5,color=colorgfdl,alpha=1,zorder=3,clip_on=False,label=r'\textbf{GFDL-CM3}',
         linestyle='--',dashes=(1,0.3))

plt.yticks(np.arange(0,101,10),map(str,np.round(np.arange(0,101,10),2)),size=5)
plt.xticks(np.arange(1950,2030+1,10),map(str,np.arange(1950,2030+1,10)),size=5)
plt.xlim([1950,2020])   
plt.ylim([0,100])  

plt.text(1948,104,r'\textbf{[d]}',color='dimgrey',
         fontsize=7,ha='center')  
leg = plt.legend(shadow=False,fontsize=9,loc='upper center',
              bbox_to_anchor=(0.5,1.23),fancybox=True,ncol=4,frameon=False,
              handlelength=5,handletextpad=1)
plt.ylabel(r'\textbf{Frequency of Label}',color='k',fontsize=7) 
plt.text(2024,50,r'\textbf{ARCTIC$_{RM}$}',color='dimgrey',fontsize=15,rotation=270,
         ha='center',va='center')          
        
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.savefig(directoryfigure + 'MSFigure-4_v7.png',dpi=1000)