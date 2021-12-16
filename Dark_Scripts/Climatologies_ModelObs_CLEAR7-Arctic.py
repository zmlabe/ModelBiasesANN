"""
Script for plotting differences in model and observational climatologies for 
select variables over the 1950 to 2019 period

Author     : Zachary M. Labe
Date       : 2 December 2021
Version    : 6 - standardizes observations by training data (only 7 models)
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
from netCDF4 import Dataset

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Read in data
directorydata = '/Users/zlabe/Documents/Projects/ModelBiasesANN/Data/'
directoryfigure = '/Users/zlabe/Documents/Projects/ModelBiasesANN/Dark_Figures/'
data = Dataset(directorydata + 'climatology_0.25g_ea_2t_11_1991-2020_v02.nc')
lats = data.variables['g0_lat_0'][:]
lons = data.variables['g0_lon_1'][:]
climall = data.variables['2T_GDS0_SFC_S123'][:]-273.15
data.close()

###############################################################################
###############################################################################
###############################################################################     
### Create plot             

fig = plt.figure()
var = climall

ax1 = plt.subplot(111)
m = Basemap(projection='npstere',boundinglat=61.5,lon_0=0,
            resolution='l',round =True,area_thresh=10000)
m.drawcoastlines(color='dimgrey',linewidth=0.27)
    
var, lons_cyclic = addcyclic(var, lons)
var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
x, y = m(lon2d, lat2d)
   
circle = m.drawmapboundary(fill_color='k',color='k',
                  linewidth=0.7)
circle.set_clip_on(False)
cs1 = m.contourf(x,y,var,np.arange(-32,15.01,0.1),extend='both')
cs1.set_cmap('twilight') 

plt.tight_layout()
plt.savefig(directoryfigure + 'Arctic_Composite_CLEAR.png',dpi=1000)
    