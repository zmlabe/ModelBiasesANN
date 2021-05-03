"""
ANN for evaluating model biases, differences, and other thresholds using 
explainable AI

Reference  : Barnes et al. [2020, JAMES]
Author     : Zachary M. Labe
Date       : 29 April 2021
Version    : 4.1 - uses 9 classes (MMean + random noise)
"""

### Import packages
import sys
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
from keras.layers import Dense, Activation
from keras import regularizers
from keras import metrics
from keras import optimizers
from keras.models import Sequential
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import random
import scipy.stats as stats
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import cmocean as cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import calc_LRPclass as LRP
import innvestigate
from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

### Prevent tensorflow 2.+ deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

### LRP param
DEFAULT_NUM_BWO_ITERATIONS = 200
DEFAULT_BWO_LEARNING_RATE = .001

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
directorydataLLL = '/Users/zlabe/Data/LENS/monthly'
directorydataENS = '/Users/zlabe/Data/SMILE/'
directorydataBB = '/Users/zlabe/Data/BEST/'
directorydataEE = '/Users/zlabe/Data/ERA5/'
directoryoutput = '/Users/zlabe/Documents/Research/ModelComparison/Data/'
###############################################################################
###############################################################################
modelGCMs = ['CCCma_canesm2','MPI','CSIRO_MK3.6','KNMI_ecearth',
              'GFDL_CM3','GFDL_ESM2M','lens']
datasetsingle = ['SMILE']
dataset_obs = 'ERA5BE'
seasons = ['annual']
variq = 'P'
reg_name = 'SMILEGlobe'
timeper = 'historical'
###############################################################################
###############################################################################
pickSMILE = []
if len(pickSMILE) >= 1:
    lenOfPicks = len(pickSMILE)
else:
    lenOfPicks = len(modelGCMs)
###############################################################################
###############################################################################
land_only = False
ocean_only = False
if land_only == True:
    maskNoiseClass = 'land'
elif ocean_only == True:
    maskNoiseClass = 'ocean'
else:
    maskNoiseClass = 'none'

###############################################################################
###############################################################################
rm_merid_mean = False
rm_annual_mean = True
###############################################################################
###############################################################################
rm_ensemble_mean = False
rm_observational_mean = False
###############################################################################
###############################################################################
calculate_anomalies = False
if calculate_anomalies == True:
    if timeper == 'historical': 
        baseline = np.arange(1951,1980+1,1)
    elif timeper == 'future':
        baseline = np.arange(2021,2050+1,1)
    else:
        print(ValueError('WRONG TIMEPER!'))
###############################################################################
###############################################################################
window = 0
ensTypeExperi = 'ENS'
# shuffletype = 'TIMEENS'
# shuffletype = 'ALLENSRAND'
# shuffletype = 'ALLENSRANDrmmean'
shuffletype = 'RANDGAUSS'
sizeOfTwinq = 2 # Name of experiment for adding noise classs #8 & #9
    
###############################################################################
###############################################################################
if ensTypeExperi == 'ENS':
    if window == 0:
        rm_standard_dev = False
        if timeper == 'historical': 
            yearsall = np.arange(1950,2019+1,1)
        elif timeper == 'future':
            yearsall = np.arange(2020,2099+1,1)
        else:
            print(ValueError('WRONG TIMEPER!'))
            sys.exit()
        ravel_modelens = False
        ravelmodeltime = False
    else:
        rm_standard_dev = True
        if timeper == 'historical': 
            yearsall = np.arange(1950+window,2019+1,1)
        elif timeper == 'future':
            yearsall = np.arange(2020+window,2099+1,1)
        else:
            print(ValueError('WRONG TIMEPER!'))
            sys.exit()
        ravelmodeltime = False
        ravel_modelens = True
elif ensTypeExperi == 'GCM':
    if window == 0:
        rm_standard_dev = False
        yearsall = np.arange(1950,2019+1,1)
        ravel_modelens = False
        ravelmodeltime = False
    else:
        rm_standard_dev = True
        if timeper == 'historical': 
            yearsall = np.arange(1950,2019+1,1)
        elif timeper == 'future':
            yearsall = np.arange(2020,2099+1,1)
        else:
            print(ValueError('WRONG TIMEPER!'))
            sys.exit()
        ravelmodeltime = False
        ravel_modelens = True
###############################################################################
###############################################################################
numOfEns = 16
lensalso = True
if len(pickSMILE) == 0:
    if modelGCMs[-1] == 'RANDOM':
        randomalso = True
    else:
        randomalso = False
elif len(pickSMILE) != 0:
    if pickSMILE[-1] == 'RANDOM':
        randomalso = True
    else:
        randomalso = False
lentime = len(yearsall)
###############################################################################
###############################################################################
ravelyearsbinary = False
ravelbinary = False
num_of_class = lenOfPicks + sizeOfTwinq
###############################################################################
###############################################################################
lrpRule = 'z'
normLRP = True
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Picking experiment to save
typeOfAnalysis = 'issueWithExperiment'

# Experiment #1
if rm_ensemble_mean == True:
    if window > 1:
        if calculate_anomalies == False:
            if rm_merid_mean == False:
                if rm_observational_mean == False:
                    if rm_annual_mean == False:
                        typeOfAnalysis = 'Experiment-1'
# Experiment #2
if rm_ensemble_mean == True:
    if window == 0:
        if calculate_anomalies == False:
            if rm_merid_mean == False:
                if rm_observational_mean == False:
                    if rm_annual_mean == False:
                        typeOfAnalysis = 'Experiment-2'
# Experiment #3 (raw data)
if rm_ensemble_mean == False:
    if window == 0:
        if calculate_anomalies == False:
            if rm_merid_mean == False:
                if rm_observational_mean == False:
                    if rm_annual_mean == False:
                        typeOfAnalysis = 'Experiment-3'
                        if variq == 'T2M':
                            integer = 2 # random noise value to add/subtract from each grid point
                        elif variq == 'P':
                            integer = 2 # random noise value to add/subtract from each grid point
                        elif variq == 'SLP':
                            integer = 2 # random noise value to add/subtract from each grid point
# Experiment #4
if rm_ensemble_mean == False:
    if window == 0:
        if calculate_anomalies == False:
            if rm_merid_mean == False:
                if rm_observational_mean == False:
                    if rm_annual_mean == True:
                        typeOfAnalysis = 'Experiment-4'
                        if variq == 'T2M':
                            integer = 2 # random noise value to add/subtract from each grid point
                        elif variq == 'P':
                            integer = 2 # random noise value to add/subtract from each grid point
                        elif variq == 'SLP':
                            integer = 2 # random noise value to add/subtract from each grid point
# Experiment #5
if rm_ensemble_mean == False:
    if window == 0:
        if calculate_anomalies == False:
            if rm_merid_mean == False:
                if rm_observational_mean == True:
                    if rm_annual_mean == False:
                        typeOfAnalysis = 'Experiment-5'
# Experiment #6
if rm_ensemble_mean == False:
    if window == 0:
        if calculate_anomalies == False:
            if rm_merid_mean == False:
                if rm_observational_mean == True:
                    if rm_annual_mean == True:
                        typeOfAnalysis = 'Experiment-6'
# Experiment #7
if rm_ensemble_mean == False:
    if window == 0:
        if calculate_anomalies == True:
            if rm_merid_mean == False:
                if rm_observational_mean == True:
                    if rm_annual_mean == False:
                        typeOfAnalysis = 'Experiment-7'
# Experiment #8
if rm_ensemble_mean == False:
    if window == 0:
        if calculate_anomalies == True:
            if rm_merid_mean == False:
                if rm_observational_mean == False:
                    if rm_annual_mean == False:
                        typeOfAnalysis = 'Experiment-8'
                        if variq == 'T2M':
                            integer = 1 # random noise value to add/subtract from each grid point
                        elif variq == 'P':
                            integer = 1 # random noise value to add/subtract from each grid point
                        elif variq == 'SLP':
                            integer = 5 # random noise value to add/subtract from each grid point
# Experiment #9
if rm_ensemble_mean == False:
    if window > 1:
        if calculate_anomalies == True:
            if rm_merid_mean == False:
                if rm_observational_mean == False:
                    if rm_annual_mean == False:
                        typeOfAnalysis = 'Experiment-9'
                        
print('\n<<<<<<<<<<<< Analysis == %s (%s) ! >>>>>>>>>>>>>>>\n' % (typeOfAnalysis,timeper))
if typeOfAnalysis == 'issueWithExperiment':
    sys.exit('Wrong parameters selected to analyze')
    
### Select how to save files
if land_only == True:
    saveData = timeper + '_LAND' + '_NoiseTwinSingleMODDIFrand_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
elif ocean_only == True:
    saveData = timeper + '_OCEAN' + '_NoiseTwinSingleMODDIFrand_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
else:
    saveData = timeper + '_NoiseTwinSingleMODDIFrand_' + typeOfAnalysis + '_' + variq + '_' + reg_name + '_' + dataset_obs + '_' + 'NumOfSMILE-' + str(num_of_class) + '_Method-' + ensTypeExperi
print('*Filename == < %s >' % saveData) 
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Create sample class labels for each model for my own testing
### Appends a twin set of classes for the random noise class 
if seasons != 'none':
    classesl = np.empty((lenOfPicks,numOfEns,len(yearsall)))
    for i in range(lenOfPicks):
        classesl[i,:,:] = np.full((numOfEns,len(yearsall)),i)  
        
    ### Add random noise models
    randomNoiseClass = np.full((sizeOfTwinq,numOfEns,len(yearsall)),i+1)
    if sizeOfTwinq > 1:
        randomNoiseClass[sizeOfTwinq-1,:,:] = i+sizeOfTwinq    
    classesl = np.append(classesl,randomNoiseClass,axis=0)
        
    if ensTypeExperi == 'ENS':
        classeslnew = np.swapaxes(classesl,0,1)
    elif ensTypeExperi == 'GCM':
        classeslnew = classesl
###############################################################################
###############################################################################
###############################################################################
###############################################################################    
### Begin ANN and the entire script
for sis,singlesimulation in enumerate(datasetsingle):
    lrpsns = []
    for seas in range(len(seasons)):
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### ANN preliminaries
        simuqq = datasetsingle[0]
        monthlychoice = seasons[seas]
        lat_bounds,lon_bounds = UT.regions(reg_name)
        directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/'
        experiment_result = pd.DataFrame(columns=['actual iters','hiddens','cascade',
                                                  'RMSE Train','RMSE Test',
                                                  'ridge penalty','zero mean',
                                                  'zero merid mean','land only?','ocean only?'])    
        
        ### Define primary dataset to use
        dataset = singlesimulation
        modelType = dataset
        
        ### Whether to test and plot the results using obs data
        if dataset_obs == '20CRv3':
            year_obsall = np.arange(yearsall[sis].min(),2015+1,1)
        elif dataset_obs == 'ERA5':
            year_obsall = np.arange(1979+window,2019+1,1)
            if rm_standard_dev == False:
                year_obsall = np.arange(1979,2019+1,1)
        elif dataset_obs == 'ERA5BE':
            year_obsall = np.arange(1950+window,2019+1,1)
            if rm_standard_dev == False:
                year_obsall = np.arange(1950,2019+1,1)
        if monthlychoice == 'DJF':
            obsyearstart = year_obsall.min()+1
            year_obs = year_obsall[1:]
        else:
            obsyearstart = year_obsall.min()
            year_obs = year_obsall
        
        ### Remove the annual mean? True to subtract it from dataset ##########
        if rm_annual_mean == True:
            directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/'
        
        ### Rove the ensemble mean? True to subtract it from dataset ##########
        if rm_ensemble_mean == True:
            directoryfigure = '/Users/zlabe/Desktop/ModelComparison_v1/'
        
        ### Split the data into training and testing sets? value of 1 will use all 
        ### data as training
        segment_data_factor = .75
        
        ### Hiddens corresponds to the number of hidden layers the nnet will use - 0 
        ### for linear model, or a list [10, 20, 5] for multiple layers of nodes 
        ### (10 nodes in first layer, 20 in second, etc); The "loop" part 
        ### allows you to loop through multiple architectures. For example, 
        ### hiddens_loop = [[2,4],[0],[1 1 1]] would produce three separate NNs, the 
        ### first with 2 hidden layers of 2 and 4 nodes, the next the linear model,
        ### and the next would be 3 hidden layers of 1 node each.
        
        ### Set useGPU to True to use the GPU, but only if you selected the GPU 
        ### Runtime in the menu at the top of this page
        useGPU = False
        
        ### Set Cascade to True to utilize the nnet's cascade function
        cascade = False
        
        ### Plot within the training loop - may want to set to False when testing out 
        ### larget sets of parameters
        plot_in_train = False
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Read in model and observational/reanalysis data
        
        def read_primary_dataset(variq,dataset,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
            data,lats,lons = df.readFiles(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
            datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
            print('\nOur dataset: ',dataset,' is shaped',data.shape)
            return datar,lats,lons
          
        def read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
            data_obs,lats_obs,lons_obs = df.readFiles(variq,dataset_obs,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
            data_obs,lats_obs,lons_obs = df.getRegion(data_obs,lats_obs,lons_obs,
                                                    lat_bounds,lon_bounds)
            
            print('our OBS dataset: ',dataset_obs,' is shaped',data_obs.shape)
            return data_obs,lats_obs,lons_obs
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Select data to test, train on           
        def segment_data(data,classesl,ensTypeExperi,fac = segment_data_factor):
          
            global random_segment_seed,trainIndices,testIndices
            if random_segment_seed == None:
                random_segment_seed = int(int(np.random.randint(1, 100000)))
            np.random.seed(random_segment_seed)

############################################################################### 
############################################################################### 
###############################################################################             
            ###################################################################
            ### Large Ensemble experiment
            if ensTypeExperi == 'ENS':
                
                ### Flip GCM and ensemble member axes
                datanew = np.swapaxes(data,0,1)
                classeslnew = np.swapaxes(classesl,0,1)
        
                if fac < 1 :
                    nrows = datanew.shape[0]
                    segment_train = int(np.round(nrows * fac))
                    segment_test = nrows - segment_train
                    print('Training on',segment_train,'ensembles, testing on',segment_test)
            
                    ### Picking out random ensembles
                    i = 0
                    trainIndices = list()
                    while i < segment_train:
                        line = np.random.randint(0, nrows)
                        if line not in trainIndices:
                            trainIndices.append(line)
                            i += 1
                        else:
                            pass
                
                    i = 0
                    testIndices = list()
                    while i < segment_test:
                        line = np.random.randint(0, nrows)
                        if line not in trainIndices:
                            if line not in testIndices:
                                testIndices.append(line)
                                i += 1
                        else:
                            pass
                        
                    ### Training segment----------
                    data_train = np.empty((len(trainIndices),datanew.shape[1],
                                            datanew.shape[2],datanew.shape[3],
                                            datanew.shape[4]))
                    Ytrain = np.empty((len(trainIndices),classeslnew.shape[1],
                                        classeslnew.shape[2]))
                    for index,ensemble in enumerate(trainIndices):
                        data_train[index,:,:,:,:] = datanew[ensemble,:,:,:,:]
                        Ytrain[index,:,:] = classeslnew[ensemble,:,:]
                        
                    ### Random ensembles are picked
                    if debug:
                        print('\nTraining on ensembles: ',trainIndices)
                        print('Testing on ensembles: ',testIndices)
                        print('\norg data - shape', datanew.shape)
                        print('training data - shape', data_train.shape)
                
                    ### Reshape into X and Y
                    Xtrain = data_train.reshape((data_train.shape[0]*data_train.shape[1]*data_train.shape[2]),(data_train.shape[3]*data_train.shape[4]))
                    Ytrain = Ytrain.reshape((Ytrain.shape[0]*Ytrain.shape[1]*Ytrain.shape[2]))
                    Xtrain_shape = (data_train.shape[0])
                            
                    ### Testing segment----------
                    data_test = np.empty((len(testIndices),datanew.shape[1],
                                            datanew.shape[2],datanew.shape[3],
                                            datanew.shape[4]))
                    Ytest = np.empty((len(testIndices),classeslnew.shape[1],
                                        classeslnew.shape[2]))
                    for index,ensemble in enumerate(testIndices):
                        data_test[index,:,:,:,:] = datanew[ensemble,:,:,:,:]
                        Ytest[index,:,:] = classeslnew[ensemble,:,:]
                    
                    ### Random ensembles are picked
                    if debug:
                        print('Training on ensembles: %s' % len(trainIndices))
                        print('Testing on ensembles: %s' % len(testIndices))
                        print('\norg data - shape', datanew.shape)
                        print('testing data - shape', data_test.shape)
        
                    ### Reshape into X and Y
                    Xtest= data_test.reshape((data_test.shape[0]*data_test.shape[1]*data_test.shape[2]),(data_test.shape[3]*data_test.shape[4]))
                    Ytest = Ytest.reshape((Ytest.shape[0]*Ytest.shape[1]*Ytest.shape[2]))
                    Xtest_shape = (data_test.shape[0])
            
                Xtest_shape = (data_test.shape[0], data_test.shape[1])
                data_train_shape = data_train.shape[0]
                data_test_shape = data_test.shape[0]
              
                ### 'unlock' the random seed
                np.random.seed(None)
                
                ### One-hot vectors
                Ytrain = keras.utils.to_categorical(Ytrain)
                Ytest = keras.utils.to_categorical(Ytest)  
                
                ### Class weights
                class_weight = class_weight_creator(Ytrain)

###############################################################################
############################################################################### 
###############################################################################                 
            ###################################################################
            ### GCM type experiments without ensembles
            elif ensTypeExperi == 'GCM':
                if data.ndim == 5:
                    datanew = np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2],data.shape[3],data.shape[4]))
                    classeslnew = np.reshape(classesl,(classesl.shape[0]*classesl.shape[1],classesl.shape[2]))
                else:
                    datanew = data
                    classeslnew = classesl
        
                if fac < 1 :
                    nrows = datanew.shape[1]
                    segment_train = int(np.floor(nrows * fac))
                    segment_test = nrows - segment_train
                    print('Training on',segment_train,'years, testing on',segment_test)
            
                    ### Picking out random ensembles
                    firstyears = int(np.floor(segment_test/2))
                    lastyears = -int(np.floor(segment_test/2))
                    trainIndices = np.arange(firstyears,firstyears+segment_train,1)
                    testIndices = np.append(np.arange(firstyears),np.arange(trainIndices[-1]+1,nrows,1),axis=0)
     
                    ### Training segment----------
                    data_train = np.empty((datanew.shape[0],len(trainIndices),
                                            datanew.shape[2],datanew.shape[3]))
                    Ytrain = np.empty((classeslnew.shape[0],len(trainIndices)))
                    for index,ensemble in enumerate(trainIndices):
                        data_train[:,index,:,:] = datanew[:,ensemble,:,:]
                        Ytrain[:,index] = classeslnew[:,ensemble]
                        
                    ### Random ensembles are picked
                    if debug:
                        print('\nTraining on years: ',trainIndices)
                        print('Testing on years: ',testIndices)
                        print('\norg data - shape', datanew.shape)
                        print('training data - shape', data_train.shape)
                
                    ### Reshape into X and Y
                    Xtrain = data_train.reshape((data_train.shape[0]*data_train.shape[1]),(data_train.shape[2]*data_train.shape[3]))
                    Ytrain = Ytrain.reshape((Ytrain.shape[0]*Ytrain.shape[1]))
                    Xtrain_shape = (data_train.shape[0])
                            
                    ### Testing segment----------
                    data_test = np.empty((datanew.shape[0],len(testIndices),
                                            datanew.shape[2],datanew.shape[3]))
                    Ytest = np.empty((classeslnew.shape[0],len(testIndices)))
                    for index,ensemble in enumerate(testIndices):
                        data_test[:,index,:,:] = datanew[:,ensemble,:,:]
                        Ytest[:,index] = classeslnew[:,ensemble]
                    
                    ### Random ensembles are picked
                    if debug:
                        print('Training on years: %s' % len(trainIndices))
                        print('Testing on years: %s' % len(testIndices))
                        print('\norg data - shape', datanew.shape)
                        print('testing data - shape', data_test.shape)
        
                    ### Reshape into X and Y
                    Xtest= data_test.reshape((data_test.shape[0]*data_test.shape[1]),(data_test.shape[2]*data_test.shape[3]))
                    Ytest = Ytest.reshape((Ytest.shape[0]*Ytest.shape[1]))
                    Xtest_shape = (data_test.shape[0])
            
                Xtest_shape = (data_test.shape[0], data_test.shape[1])
                data_train_shape = data_train.shape[0]
                data_test_shape = data_test.shape[0]
              
                ### 'unlock' the random seed
                np.random.seed(None)
                
                ### One-hot vectors
                Ytrain = keras.utils.to_categorical(Ytrain)
                Ytest = keras.utils.to_categorical(Ytest) 
                
                ### Class weights
                class_weight = class_weight_creator(Ytrain)
          
            else:
                print(ValueError('WRONG EXPERIMENT!'))
            return Xtrain,Ytrain,Xtest,Ytest,Xtest_shape,Xtrain_shape,data_train_shape,data_test_shape,testIndices,trainIndices,class_weight
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Plotting functions   
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
        ### Create a class weight dictionary to help if the classes are unbalanced
        def class_weight_creator(Y):
            class_dict = {}
            weights = np.max(np.sum(Y, axis=0)) / np.sum(Y, axis=0)
            for i in range( Y.shape[-1] ):
                class_dict[i] = weights[i]               
            return class_dict
                    
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Neural Network Creation & Training        
        class TimeHistory(keras.callbacks.Callback):
            def on_train_begin(self, logs={}):
                self.times = []
        
            def on_epoch_begin(self, epoch, logs={}):
                self.epoch_time_start = time.time()
        
            def on_epoch_end(self, epoch, logs={}):
                self.times.append(time.time() - self.epoch_time_start)
        
        def defineNN(hidden, input_shape, output_shape, ridgePenalty):        
           
            model = Sequential()
            ### Initialize first layer
            ### Model is a single node with activation function
            model.add(Dense(hidden[0],input_shape=(input_shape,),
                            activation=actFun, use_bias=True,
                            kernel_regularizer=regularizers.l1_l2(l1=0.00,l2=ridgePenalty),
                            bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                            kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
    
            ### Initialize other layers
            for layer in hidden[1:]:
                model.add(Dense(layer,activation=actFun,
                                use_bias=True,
                                kernel_regularizer=regularizers.l1_l2(l1=0.00,l2=0.00),
                                bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                                kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
                    
                print('\nTHIS IS AN ANN!\n')
        
            #### Initialize output layer
            model.add(Dense(output_shape,activation=None,use_bias=True,
                            kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.00),
                            bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                            kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
        
            ### Add softmax layer at the end
            model.add(Activation('softmax'))
            
            return model
        
        def trainNN(model, Xtrain, Ytrain, niter, class_weight, verbose):
          
            global lr_here, batch_size
            lr_here = 0.001
            model.compile(optimizer=optimizers.SGD(lr=lr_here,
                          momentum=0.9,nesterov=True),  
                          loss = 'categorical_crossentropy',
                          metrics=[metrics.categorical_accuracy])
            # model.compile(optimizer=optimizers.Nadam(lr=lr_here),  
            #               loss = 'categorical_crossentropy',
            #               metrics=[metrics.categorical_accuracy])
        
            ### Declare the relevant model parameters
            batch_size = 24 
        
            print('----ANN Training: learning rate = '+str(lr_here)+'; activation = '+actFun+'; batch = '+str(batch_size) + '----')    
            
            ### Callbacks
            time_callback = TimeHistory()
            early_stopping = keras.callbacks.EarlyStopping(monitor='loss',
                                                           patience=2,
                                                           verbose=1,
                                                           mode='auto')
            
            history = model.fit(Xtrain,Ytrain,batch_size=batch_size,epochs=niter,
                                shuffle=True,verbose=verbose,
                                callbacks=[time_callback,early_stopping],
                                validation_split=0.)
            print('******** done training ***********')
        
            return model, history
        
        def test_train_loopClass(Xtrain,Ytrain,Xtest,Ytest,iterations,ridge_penalty,hiddens,class_weight,plot_in_train=True):
            """or loops to iterate through training iterations, ridge penalty, 
            and hidden layer list
            """
            results = {}
            global nnet,random_network_seed
          
            for niter in iterations:
                for penalty in ridge_penalty:
                    for hidden in hiddens:
                        
                        ### Check / use random seed
                        if random_network_seed == None:
                          np.random.seed(None)
                          random_network_seed = int(np.random.randint(1, 100000))
                        np.random.seed(random_network_seed)
                        random.seed(random_network_seed)
                        tf.set_random_seed(0)
        
                        ### Standardize the data
                        Xtrain,Xtest,stdVals = dSS.standardize_data(Xtrain,Xtest)
                        Xmean,Xstd = stdVals
                        
                        ### Define the model
                        model = defineNN(hidden,
                                          input_shape=np.shape(Xtrain)[1],
                                          output_shape=np.shape(Ytrain)[1],
                                          ridgePenalty=penalty)  
                       
                        ### Train the net
                        model, history = trainNN(model,Xtrain,
                                                  Ytrain,niter,class_weight,verbose=1)
        
                        ### After training, use the network with training data to 
                        ### check that we don't have any errors and output RMSE
                        rmse_train = dSS.rmse(Ytrain,model.predict(Xtrain))
                        if type(Ytest) != bool:
                            rmse_test = 0.
                            rmse_test = dSS.rmse(Ytest,model.predict(Xtest))
                        else:
                            rmse_test = False
        
                        this_result = {'iters': niter, 
                                        'hiddens' : hidden, 
                                        'RMSE Train' : rmse_train, 
                                        'RMSE Test' : rmse_test, 
                                        'ridge penalty': penalty, 
                                        'zero mean' : rm_annual_mean,
                                        'zero merid mean' : rm_merid_mean,
                                        'land only?' : land_only,
                                        'ocean only?' : ocean_only,
                                        'Segment Seed' : random_segment_seed,
                                        'Network Seed' : random_network_seed }
                        results.update(this_result)
        
                        global experiment_result
                        experiment_result = experiment_result.append(results,
                                                                      ignore_index=True)
                        
                        #if True to plot each iter's graphs.
                        if plot_in_train == True:
                            plt.figure()
        
                            plt.subplot(1,1,1)
                            plt.plot(history.history['loss'],label = 'training')
                            plt.title(history.history['loss'][-1])
                            plt.xlabel('epoch')
                            plt.xlim(2,len(history.history['loss'])-1)
                            plt.legend()
                            
                            plt.grid(True)
                            plt.show()
        
                        #'unlock' the random seed
                        np.random.seed(None)
                        random.seed(None)
                        tf.set_random_seed(None)
          
            return experiment_result, model
        
        ###############################################################################
        ###############################################################################
        ###############################################################################
        ### Results
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1)
        
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
        K.clear_session()
        
        ### Parameters
        debug = True
        NNType = 'ANN'
        avgHalfChunk = 0
        option4 = True
        biasBool = False
        hiddensList = [[10,10]]
        ridge_penalty = [0.10]
        actFun = 'relu'
        
        if any([maskNoiseClass=='land',maskNoiseClass=='ocean']):
            debug = True
            NNType = 'ANN'
            avgHalfChunk = 0
            option4 = True
            biasBool = False
            hiddensList = [[8,8]]
            ridge_penalty = [0.10]
            actFun = 'relu'
        
        expList = [(0)] # (0,1)
        expN = np.size(expList)
        
        iterations = [100] 
        random_segment = True
        foldsN = 1
        
        for avgHalfChunk in (0,): # ([1,5,10]):#([1,2,5,10]):
            session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                          inter_op_parallelism_threads=1)
            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
            K.set_session(sess)
            K.clear_session()
            
            for loop in ([0]): # (0,1,2,3,4,5):
                ### Get info about the region
                lat_bounds,lon_bounds = UT.regions(reg_name)
                data_all,lats,lons = read_primary_dataset(variq,dataset,
                                                          numOfEns,lensalso,
                                                          randomalso,
                                                          ravelyearsbinary,
                                                          ravelbinary,
                                                          shuffletype,
                                                          lat_bounds,
                                                          lon_bounds)
                data_obs_all,lats_obs,lons_obs = read_obs_dataset(variq,
                                                                  dataset_obs,
                                                                  numOfEns,
                                                                  lensalso,
                                                                  randomalso,
                                                                  ravelyearsbinary,
                                                                  ravelbinary,
                                                                  shuffletype,
                                                                  lat_bounds,
                                                                  lon_bounds)

###############################################################################
###############################################################################
###############################################################################
                for exp in expList:                      
                    ### Get the data together
                    data, data_obs, = data_all, data_obs_all,
###############################################################################
                    if len(pickSMILE) >= 1:
                        data = dSS.pickSmileModels(data,modelGCMs,pickSMILE)
                        print('\n*Pick models to analysis from %s*\n' % pickSMILE)
###############################################################################
                    if calculate_anomalies == True:
                        data, data_obs = dSS.calculate_anomalies(data,data_obs,
                                          lats,lons,baseline,yearsall)
                        print('\n*Calculate anomalies for %s-%s*\n' % (baseline.min(),baseline.max()))
###############################################################################                        
                    if rm_annual_mean == True:
                        data, data_obs = dSS.remove_annual_mean(data,data_obs,
                                                            lats,lons,
                                                            lats_obs,lons_obs)
                        print('\n*Removed annual mean*\n')
###############################################################################                        
                    if rm_merid_mean == True:
                        data, data_obs = dSS.remove_merid_mean(data,data_obs,
                                                            lats,lons,
                                                            lats_obs,lons_obs)
                        print('\n*Removed meridional mean*\n')
###############################################################################                        
                    if rm_ensemble_mean == True:
                        data = dSS.remove_ensemble_mean(data,ravel_modelens,
                                                          ravelmodeltime,
                                                          rm_standard_dev,
                                                          numOfEns)
                        print('\n*Removed ensemble mean*')
###############################################################################                       
                    if rm_standard_dev == True:
                        data = dSS.rm_standard_dev(data,window,ravelmodeltime,
                                                    numOfEns)
                        print('\n*Removed standard deviation*')
###############################################################################                        
                    if rm_observational_mean == True:
                        data = dSS.remove_observations_mean(data,data_obs,lats,lons)
                        print('\n*Removed observational data*')
###############################################################################                        
                    if land_only == True:
                        data, data_obs = dSS.remove_ocean(data,data_obs,
                                                          lat_bounds,
                                                          lon_bounds) 
                        print('\n*Removed ocean data*')
###############################################################################
                    if ocean_only == True:
                        data, data_obs = dSS.remove_land(data,data_obs,
                                                          lat_bounds,
                                                          lon_bounds) 
                        print('\n*Removed land data*')     
###############################################################################
                    ### Adding random data
                    if sizeOfTwinq > 0:
                        random_segment_seed = int(np.genfromtxt('/Users/zlabe/Documents/Research/ModelComparison/Data/SelectedSegmentSeed.txt',unpack=True))
                        data = dSS.addNoiseTwinSingle(data,data_obs,integer,4,random_segment_seed,maskNoiseClass,lat_bounds,lon_bounds)
                        
                        random_segment_seed = int(np.genfromtxt('/Users/zlabe/Documents/Research/ModelComparison/Data/SelectedSegmentSeed.txt',unpack=True))
                        data = dSS.addNoiseTwinSingle(data,data_obs,integer,6,random_segment_seed,maskNoiseClass,lat_bounds,lon_bounds)
###############################################################################
###############################################################################
###############################################################################
                    ### Loop over folds
                    for loop in np.arange(0,foldsN): 
        
                        K.clear_session()
                        #---------------------------
                        # random_segment_seed = 34515
                        random_segment_seed = int(np.genfromtxt('/Users/zlabe/Documents/Research/ModelComparison/Data/SelectedSegmentSeed.txt',unpack=True))
                        #---------------------------
                        Xtrain,Ytrain,Xtest,Ytest,Xtest_shape,Xtrain_shape,data_train_shape,data_test_shape,testIndices,trainIndices,class_weight = segment_data(data,classesl,ensTypeExperi,segment_data_factor)
        
                        YtrainClassMulti = Ytrain  
                        YtestClassMulti = Ytest  
        
                        # For use later
                        XtrainS,XtestS,stdVals = dSS.standardize_data(Xtrain,Xtest)
                        Xmean, Xstd = stdVals      
        
                        #---------------------------
                        random_network_seed = 87750
                        #---------------------------
        
                        # Create and train network
                        exp_result,model = test_train_loopClass(Xtrain,
                                                                YtrainClassMulti,
                                                                Xtest,
                                                                YtestClassMulti,
                                                                iterations=iterations,
                                                                ridge_penalty=ridge_penalty,
                                                                hiddens=hiddensList,class_weight=class_weight,
                                                                plot_in_train = True)
                        model.summary()  
                        
                        ################################################################################################################################################                
                        # save the model
                        dirname = '/Users/zlabe/Desktop/ModelComparison_v1/'
                        savename = modelType+'_'+variq+'_kerasMultiClassBinaryOption4'+'_' + NNType + '_L2_'+ str(ridge_penalty[0])+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(iterations[0]) + '_' + str(hiddensList[0][0]) + 'x' + str(hiddensList[0][-1]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
                        savenameModelTestTrain = modelType+'_'+variq+'_modelTrainTest_SegSeed'+str(random_segment_seed)+'_NetSeed'+str(random_network_seed)
        
                        if(reg_name=='Globe'):
                            regSave = ''
                        else:
                            regSave = '_' + reg_name
                        
                        if(rm_annual_mean==True):
                            savename = savename + '_AnnualMeanRemoved' 
                            savenameModelTestTrain = savenameModelTestTrain + '_AnnualMeanRemoved'
                        if(rm_ensemble_mean==True):
                            savename = savename + '_EnsembleMeanRemoved' 
                            savenameModelTestTrain = savenameModelTestTrain + '_EnsembleMeanRemoved'
        
                        savename = savename + regSave    
                        # model.save(dirname + savename + '.h5')
                        # np.savez(dirname + savenameModelTestTrain + '.npz',trainModels=trainIndices,testModels=testIndices,Xtrain=Xtrain,Ytrain=Ytrain,Xtest=Xtest,Ytest=Ytest,Xmean=Xmean,Xstd=Xstd,lats=lats,lons=lons)
        
                        print('saving ' + savename)
                        
                        ###############################################################
                        ### Make final plot
                        ### Get obs
                        dataOBSERVATIONS = data_obs
                        latsOBSERVATIONS = lats_obs
                        lonsOBSERVATIONS = lons_obs
        
                        Xobs = dataOBSERVATIONS.reshape(dataOBSERVATIONS.shape[0],dataOBSERVATIONS.shape[1]*dataOBSERVATIONS.shape[2])
        
                        annType = 'class'
                        if monthlychoice == 'DJF':
                            startYear = yearsall[sis].min()+1
                            endYear = yearsall[sis].max()
                        else:
                            startYear = yearsall[sis].min()
                            endYear = yearsall[sis].max()
                        years = np.arange(startYear,endYear+1,1)    
                        Xmeanobs = np.nanmean(Xobs,axis=0)
                        Xstdobs = np.nanstd(Xobs,axis=0)  
                        
                        XobsS = (Xobs-Xmeanobs)/Xstdobs
                        XobsS[np.isnan(XobsS)] = 0
                        
                        xtrainpred = (Xtrain-Xmean)/Xstd
                        xtrainpred[np.isnan(xtrainpred)] = 0
                        xtestpred = (Xtest-Xmean)/Xstd
                        xtestpred[np.isnan(xtestpred)] = 0
                        
                        if(annType=='class'):
                            YpredObs = model.predict(XobsS)
                            YpredTrain = model.predict(xtrainpred)
                            YpredTest = model.predict(xtestpred)
                        
                        #######################################################
                        #######################################################
                        #######################################################
                        ### Check null hypothesis of random data!
                        randarray,latsra,lonsra = read_primary_dataset(variq,'RANDOM',
                                                                       numOfEns,lensalso,
                                                                       randomalso,
                                                                       ravelyearsbinary,
                                                                       ravelbinary,
                                                                       shuffletype,
                                                                       lat_bounds,
                                                                       lon_bounds)
                        randarrayn = randarray.reshape(randarray.shape[0],randarray.shape[1]*randarray.shape[2])
                        randarraymean = np.nanmean(randarrayn,axis=0)
                        randarraystd = np.nanstd(randarrayn,axis=0)
                        randarrayS = (randarrayn-randarraymean)/randarraystd
                        
                        ### Prediction on random data
                        YpredRand = model.predict(randarrayS)
                        #######################################################
                        #######################################################
                        #######################################################
        
        ### Get output from model
        trainingout = YpredTrain
        testingout = YpredTest
        
        if ensTypeExperi == 'ENS':
            classesltrain = classeslnew[trainIndices,:,:].ravel()
            classesltest = classeslnew[testIndices,:,:].ravel()
        elif ensTypeExperi == 'GCM':
            classesltrain = classeslnew[:,:,trainIndices].ravel()
            classesltest = classeslnew[:,:,testIndices].ravel()
            
        ### Random data tests
        randout = YpredRand
        labelsrand = np.argmax(randout,axis=1)
        uniquerand,countrand = np.unique(labelsrand,return_counts=True)
        np.savetxt(directoryoutput + 'RandLabels_' + saveData + '.txt',labelsrand)
            
        ### Observations
        obsout = YpredObs
        labelsobs = np.argmax(obsout,axis=1)
        uniqueobs,countobs = np.unique(labelsobs,return_counts=True)
        np.savetxt(directoryoutput + 'obsLabels_' + saveData + '.txt',labelsobs)
        
        def truelabel(data):
            """
            Calculate argmax
            """
            maxindexdata= np.argmax(data[:,:],axis=1)    
            
            return maxindexdata
        
        def accuracyTotalTime(data_pred,data_true):
            """
            Compute accuracy for the entire time series
            """
            
            data_truer = data_true
            data_predr = data_pred
            accdata_pred = accuracy_score(data_truer,data_predr)
                
            return accdata_pred

        ##############################################################################
        ##############################################################################
        ##############################################################################        
        indextrain = truelabel(trainingout)
        acctrain = accuracyTotalTime(indextrain,classesltrain)
        indextest = truelabel(testingout)
        acctest = accuracyTotalTime(indextest,classesltest)
        print('\n\nAccuracy Training == ',acctrain)
        print('Accuracy Testing == ',acctest)
        
        ## Save the output for plotting
        np.savetxt(directoryoutput + 'trainingEnsIndices_' + saveData + '.txt',trainIndices)
        np.savetxt(directoryoutput + 'testingEnsIndices_' + saveData + '.txt',testIndices)
        
        np.savetxt(directoryoutput + 'trainingTrueLabels_' + saveData + '.txt',classesltrain)
        np.savetxt(directoryoutput + 'testingTrueLabels_' + saveData + '.txt',classesltest)
        
        np.savetxt(directoryoutput + 'trainingPredictedLabels_' + saveData + '.txt',indextrain)
        np.savetxt(directoryoutput + 'testingPredictedLabels_' + saveData + '.txt',indextest)
    
        ### See more more details
        model.layers[0].get_config()
   
        ## Define variable for analysis
        print('\n\n------------------------')
        print(variq,'= Variable!')
        print(monthlychoice,'= Time!')
        print(reg_name,'= Region!')
        print(lat_bounds,lon_bounds)
        print(dataset,'= Model!')
        print(dataset_obs,'= Observations!\n')
        print(rm_annual_mean,'= rm_annual_mean') 
        print(rm_merid_mean,'= rm_merid_mean') 
        print(rm_ensemble_mean,'= rm_ensemble_mean') 
        print(land_only,'= land_only')
        print(ocean_only,'= ocean_only')
        
        ## Variables for plotting
        lons2,lats2 = np.meshgrid(lons,lats) 
        observations = data_obs
        modeldata = data
        modeldatamean = np.nanmean(modeldata,axis=1)
        
        spatialmean_obs = UT.calc_weightedAve(observations,lats2)
        spatialmean_mod = UT.calc_weightedAve(modeldata,lats2)
        spatialmean_modmean = np.nanmean(spatialmean_mod,axis=1)
        plt.figure()
        plt.plot(spatialmean_modmean.transpose())
        
        ##############################################################################
        ##############################################################################
        ##############################################################################
        ## Visualizing through LRP
        numLats = lats.shape[0]
        numLons = lons.shape[0]  
        numDim = 3

        ##############################################################################
        ##############################################################################
        ##############################################################################
        
        lrpall = LRP.calc_LRPModel(model,np.append(XtrainS,XtestS,axis=0),
                                                np.append(Ytrain,Ytest,axis=0),
                                                biasBool,annType,num_of_class,
                                                yearsall,lrpRule,normLRP,
                                                numLats,numLons,numDim)
        meanlrp = np.nanmean(lrpall,axis=0)
        fig=plt.figure()
        plt.contourf(meanlrp,300,cmap=cmocean.cm.thermal)
        
        ### For training data only
        lrptrain = LRP.calc_LRPModel(model,XtrainS,Ytrain,biasBool,
                                                annType,num_of_class,
                                                yearsall,lrpRule,normLRP,
                                                numLats,numLons,numDim)
        
        ### For training data only
        lrptest = LRP.calc_LRPModel(model,XtestS,Ytest,biasBool,
                                                annType,num_of_class,
                                                yearsall,lrpRule,normLRP,
                                                numLats,numLons,numDim)
        
        
        ### For observations data only
        lrpobservations = LRP.calc_LRPObs(model,XobsS,biasBool,annType,
                                            num_of_class,yearsall,lrpRule,
                                            normLRP,numLats,numLons,numDim)

        ### For random data only
        lrprandom = LRP.calc_LRPObs(model,randarrayS,biasBool,annType,
                                            num_of_class,yearsall,lrpRule,
                                            normLRP,numLats,numLons,numDim)
      
        ##############################################################################
        ##############################################################################
        ##############################################################################
        def netcdfLRP(lats,lons,var,directory,typemodel,saveData):
            print('\n>>> Using netcdfLRP function!')
            
            from netCDF4 import Dataset
            import numpy as np
            
            name = 'LRPMap' + typemodel + '_' + saveData + '.nc'
            filename = directory + name
            ncfile = Dataset(filename,'w',format='NETCDF4')
            ncfile.description = 'LRP maps for using selected seed' 
            
            ### Dimensions
            ncfile.createDimension('years',var.shape[0])
            ncfile.createDimension('lat',var.shape[1])
            ncfile.createDimension('lon',var.shape[2])
            
            ### Variables
            years = ncfile.createVariable('years','f4',('years'))
            latitude = ncfile.createVariable('lat','f4',('lat'))
            longitude = ncfile.createVariable('lon','f4',('lon'))
            varns = ncfile.createVariable('LRP','f4',('years','lat','lon'))
            
            ### Units
            varns.units = 'unitless relevance'
            ncfile.title = 'LRP relevance'
            ncfile.instituion = 'Colorado State University'
            ncfile.references = 'Barnes et al. [2020]'
            
            ### Data
            years[:] = np.arange(var.shape[0])
            latitude[:] = lats
            longitude[:] = lons
            varns[:] = var
            
            ncfile.close()
            print('*Completed: Created netCDF4 File!')
        
        netcdfLRP(lats,lons,lrpall,directoryoutput,'AllData',saveData)
        netcdfLRP(lats,lons,lrptrain,directoryoutput,'Training',saveData)
        netcdfLRP(lats,lons,lrptest,directoryoutput,'Testing',saveData)
        netcdfLRP(lats,lons,lrpobservations,directoryoutput,'Obs',saveData)