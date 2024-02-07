# %%

#Import needed packages 
import xarray as xr
import matplotlib.pyplot as plt 
import numpy as np
import sys
import matplotlib
import pandas as pd
import os



#I'm having some issues with the interactive versus VS Code terminal directories so Im going to
#establish my cwd:
myScriptLocation = os.path.dirname(os.path.realpath(__file__))
os.chdir(myScriptLocation)


path_to_data = "../../datasets/sevir/"

#open an example storm from the Storm EVent ImagRy (SEVIR) dataset
ds = xr.open_dataset("onestorm.nc")

#Want to practice object-oriented programming by creating a class
#ImagePlotter that plots images and is flexible enough to plot whatever 
#variable #is given. zPutting it in an if loop because I don't want to plot every time.
plot_images=False
if plot_images:
    #Initialize the class
    from ImagePlotter import ImagePlotter
    # Create an instance of ImagePlotter
    image_plotter = ImagePlotter()
    # Assuming 'ds' is your dataset
    # Plot individual variables as needed
    # Change frame_idx to change time step. The sample file has 12 time steps
    # Could loop this to encompass every time step in the array
    image_plotter.plot_single_variable(ds, 'visible', frame_idx=0, cmap='Greys_r', colorbar_label='Reflectance factor', title='Visible (red; 0.64 $\mu$m)', scaling_factor=1e-4)
    image_plotter.plot_single_variable(ds, 'water_vapor', frame_idx=0, cmap='Blues', colorbar_label='Brightness Temperature, [$\degree$C]', title='Mid-level Water Vapor (6.9 $\mu$m)', scaling_factor=1e-2)
    image_plotter.plot_single_variable(ds, 'clean_infrared', frame_idx=0, cmap='cividis', colorbar_label='Brightness Temperature, [$\degree$C]', title='Clean Infrared (10.7 $\mu$m)', scaling_factor=1e-2)
    image_plotter.plot_single_variable(ds, 'vil', frame_idx=0, cmap='Spectral_r', colorbar_label='Vertically Integrated Liquid, [$kg \ m^{-2}$]', title='NEXRAD', scaling_factor=1.0)
    image_plotter.plot_single_variable(ds, 'lightning_flashes', frame_idx=0, cmap='magma', colorbar_label='Number of flashes', title='GLM Lightning', scaling_factor=1.0)

#number of pixels in this dataset
#print('{} is a lot of pixels'.format(ds.x.shape[0]**2 + ds.x2.shape[0]**2 + ds.x3.shape[0]**2 + ds.x4.shape[0]**2))

my_subset = ds.isel(t=0)
visible_array=my_subset.visible*1e-4

#Another pactice in creating a class, this time I'm looking to compute statistics from a given data array:
#Initialize the class:
from SevirStats import SevirStats
compute_stats = SevirStats()

print("computing statistics")
#Get needed statistics. First percentiles (defined as 0,1,10,25,50,75,90,99,100 in the class)
compute_stats.compute_percentiles(visible_array)





#load an example dataframe from a CSV file:
df = pd.read_csv('../../datasets/sevir/IR_stats_master.csv',index_col=0,low_memory=False,parse_dates=True)

#scale back to correct units
keys = list(df.keys()[:-1])
df[keys] = df[keys]*1e-2



#this is so I know how to label the plot and can be flexible based on data type
datasrc = "IR"  
plotGraph = False

print("plotGraph")

if plotGraph:
    #remove outlier
    df = df.where(df.q000 > -100)
    #make sure index is in order
    df = df.sort_index()
    df = df.dropna()


    time_range = slice('2018-10-01','2018-11-01') #we're going to use one month as an example
    df_october = df[time_range]

    df_october.index.min(),df_october.index.max()

    #Group all the data by their month
    #this is in order Jan, Feb, Mar ...
    groups = df.q000.groupby(df.index.month)
    #TODO: Could I extend this to group by season? (DJF, MAM, JJA, SON month groupings)
    from GraphPlotter import GraphPlotter
    plot_monthly = GraphPlotter()
    plot_monthly.plot_graph(groups, datasrc)
 

#make training time slice
#Going to do a random split of the data:

############  make simple y ###############
#load the label matrix, the number of GLM flashes
df_label = pd.read_csv('../../datasets/sevir/LI_stats_master.csv',index_col=0,low_memory=False,parse_dates=True)

print("Going into model training...")
#train/val/test splitting
from ModelTraining import ModelTraining
model_training = ModelTraining()
(X_train,y_train),(X_validate,y_validate),(X_test,y_test) = model_training.train_model(df, df_label, path_to_data, dropzeros=False, features_to_keep=np.arange(0,36,1),class_labels=True)

print("Got training, validation, and test data:")

print(X_train.shape, X_validate.shape, X_test.shape)

