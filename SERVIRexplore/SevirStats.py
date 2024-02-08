#Sevir stats takes the data object and conducts statistics on it

import numpy as np

class SevirStats:
    def __init__(self):
        pass  # You may not need an __init__ method for this example

    def compute_percentiles(self, data_array):
        desired_percentiles = np.array([0,1,10,25,50,75,90,99,100])
        percentiles = np.nanpercentile(data_array.values,desired_percentiles,axis=(0,1))
        #print(percentiles)


        
