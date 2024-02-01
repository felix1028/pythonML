# %%

#Import needed packages 
import xarray as xr
import matplotlib.pyplot as plt 
import numpy as np
import sys
import matplotlib


#open an example storm from the Storm EVent ImagRy (SEVIR) dataset
ds = xr.open_dataset("onestorm.nc")
#see the data by printing ds. By putting at the bottom of the cell, it is automatically printed
print(ds)

mydata = ds


from ImagePlotter import ImagePlotter

# Assuming 'ds' is your dataset
# Create an instance of ImagePlotter
image_plotter = ImagePlotter()

# Plot individual variables as needed
# Change frame_idx to change time step. The sample file has 12 time steps
image_plotter.plot_single_variable(ds, 'visible', frame_idx=0, cmap='Greys_r', colorbar_label='Reflectance factor', title='Visible (red; 0.64 $\mu$m)', scaling_factor=1e-4)
image_plotter.plot_single_variable(ds, 'water_vapor', frame_idx=0, cmap='Blues', colorbar_label='Brightness Temperature, [$\degree$C]', title='Mid-level Water Vapor (6.9 $\mu$m)', scaling_factor=1e-2)
image_plotter.plot_single_variable(ds, 'clean_infrared', frame_idx=0, cmap='cividis', colorbar_label='Brightness Temperature, [$\degree$C]', title='Clean Infrared (10.7 $\mu$m)', scaling_factor=1e-2)
image_plotter.plot_single_variable(ds, 'vil', frame_idx=0, cmap='Spectral_r', colorbar_label='Vertically Integrated Liquid, [$kg \ m^{-2}$]', title='NEXRAD', scaling_factor=1.0)
image_plotter.plot_single_variable(ds, 'lightning_flashes', frame_idx=0, cmap='magma', colorbar_label='Number of flashes', title='GLM Lightning', scaling_factor=1.0)

#number of pixels in this dataset
print('{} is a lot of pixels'.format(ds.x.shape[0]**2 + ds.x2.shape[0]**2 + ds.x3.shape[0]**2 + ds.x4.shape[0]**2))

my_subset = ds.isel(t=0)
visible_array=my_subset.visible*1e-4

desired_percentiles = np.array([0,1,10,25,50,75,90,99,100])
percentiles = np.nanpercentile(visible_array.values,desired_percentiles,axis=(0,1))

print(percentiles)




print("Here")

