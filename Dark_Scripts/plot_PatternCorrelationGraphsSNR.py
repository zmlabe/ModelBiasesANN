"""
Script for plotting pattern correlations between models and observations

Author     : Zachary M. Labe
Date       : 28 April 2021
Version    : 1
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import palettable.cubehelix as cm
import palettable.scientific.sequential as sss
import palettable.cartocolors.qualitative as cc
import cmocean as cmocean
import calc_Utilities as UT
import scipy.stats as sts

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

### Paramters
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
datasetsingle = ['SMILE']
dataset_obs = ['ERA5BE']
monthlychoiceq = ['annual','JFM','AMJ','JAS','OND']
typeOfCorr = ['R-SNR']
variables = ['T2M','P','SLP']
reg_name = 'SMILEGlobe'
level = 'surface'
timeper = 'historical'
option = 8

### Read in data
for vv in range(len(variables)):
    for mo in range(len(monthlychoiceq)):
# for vv in range(1):
#     for mo in range(1):
        variq = variables[vv]
        monthlychoice = monthlychoiceq[mo]
        directorydata = '/Users/zlabe/Documents/Research/ModelComparison/Data/Climatologies/'
        directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/Climatologies/SNR/'
        saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + timeper
        print('*Filename == < %s >' % saveData) 
        
        corrsnr = np.load(directorydata + saveData + '_corrsSNR.npz')['arr_0'][:option]
        modelGCMs = ['CanESM2','MPI','CSIRO-MK3.6','KNMI-ecearth','GFDL-CM3','GFDL-ESM2M','LENS','MMmean'][:option]
        
        ### Correlation for snr averaged across ensemble members
        correnssnr = np.nanmean(corrsnr,axis=1)
        
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
        
###############################################################################
###############################################################################
###############################################################################   
        fig = plt.figure()
        ax = plt.subplot(111)
        adjust_spines(ax, ['left', 'bottom'])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('dimgrey')
        ax.spines['bottom'].set_color('dimgrey')
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(0)
        ax.tick_params('y',length=4,width=2,which='major',color='dimgrey')
        ax.tick_params('x',length=0,width=0,which='major',color='dimgrey')
        
        xbar = np.arange(option)
        rects = plt.bar(xbar,correnssnr, align='center',zorder=2)
        
        plt.axhline(y=np.max(correnssnr),linestyle='--',linewidth=2,color='k',
                    clip_on=False,zorder=100,dashes=(1,0.3))
        plt.axhline(y=0,linestyle='-',linewidth=2,color='dimgrey',
                    clip_on=False,zorder=1)
        
        ### Set color
        colorlist = ['darkcyan']*8
        for cccc in range(option):
            rects[cccc].set_color(colorlist[cccc])
            rects[cccc].set_edgecolor(colorlist[cccc])
        
        plt.xticks(np.arange(0,option+1,1),modelGCMs,size=5.45)
        plt.yticks(np.arange(-1,1.1,0.1),map(str,np.round(np.arange(-1,1.1,0.1),2)),size=6)
        plt.xlim([-0.5,option-1+0.5])   
        plt.ylim([-0.1,1])
        
        plt.xlabel(r'\textbf{Average SNR-R - [climate model data] - %s for %s}' % (monthlychoice,variq),color='dimgrey',fontsize=8,labelpad=8)
        plt.title(r'\textbf{...AND FOR SNR}',color='k',fontsize=15)
        
        plt.tight_layout()
        plt.savefig(directoryfigure + saveData + '_%s_RawClimateModel_PatternCorrelationsSNR_%sclasses.png' % (typeOfCorr[0],len(modelGCMs)),dpi=300)