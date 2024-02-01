import matplotlib.pyplot as plt
import numpy as np

class ImagePlotter:
    def __init__(self):
        pass  # You may not need an __init__ method for this example

    def plot_single_variable(self, dataset, variable, frame_idx=0, cmap='Greys_r', vmin=None, vmax=None, colorbar_label=None, title=None, scaling_factor=1.0):
        # Extract data for the given variable and frame index
        data = getattr(dataset, variable)[:, :, frame_idx] * scaling_factor

        # Create a figure
        plt.figure(figsize=(8, 6))

        # Plot the image
        pm = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

        # Show colorbar if label is provided
        if colorbar_label:
            cbar = plt.colorbar(pm, shrink=0.7, extend='both')
            cbar.set_label(colorbar_label, fontsize=12)
            cbar.ax.tick_params(labelsize=12)

        # Set title of the plot
        if title:
            plt.title(title)

        # Display the plot
        plt.show()
        