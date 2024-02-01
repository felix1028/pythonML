import matplotlib.pyplot as plt

class PlotSubplot:
    def __init__(self):
        pass  # You may not need an __init__ method for this example

    def plot_subplot(self, ax, dataset, variable, figsize=(15, 10), frame_idx=0, cmap='Greys_r', vmin=None, vmax=None, colorbar_label=None, title=None, subplot_label=None, scaling_factor=1.0):
        # Extract data for the given variable and frame index
        data = getattr(dataset, variable)[:, :, frame_idx] * scaling_factor

        # Create a figure with specified size
        plt.figure(figsize=(15, 10))
        
        # Plot the image
        pm = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

        # Show colorbar if label is provided
        if colorbar_label:
            cbar = plt.colorbar(pm, shrink=0.7, extend='both')
            cbar.set_label(colorbar_label, fontsize=12)
            cbar.ax.tick_params(labelsize=12)

        # Set title of the subplot
        if title:
            ax.set_title(title)

        # Add subplot label
        if subplot_label:
            ax.text(0.05, 0.15, subplot_label, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.5, 'boxstyle': 'round'})

    def plot_multiple_subplots(self, dataset, frame_idx=0):
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # Example usage
        self.plot_subplot(axs[0, 0], dataset, 'visible', frame_idx, cmap='Greys_r', colorbar_label='Reflectance factor', title='Visible (red; 0.64 $\mu$m)', subplot_label='a)', scaling_factor=1e-4)
        self.plot_subplot(axs[0, 1], dataset, 'water_vapor', frame_idx, cmap='Blues', colorbar_label='Brightness Temperature, [$\degree$C]', title='Mid-level Water Vapor (6.9 $\mu$m)', subplot_label='b)', scaling_factor=1e-2)
        self.plot_subplot(axs[0, 2], dataset, 'clean_infrared', frame_idx, cmap='cividis', colorbar_label='Brightness Temperature, [$\degree$C]', title='Clean Infrared (10.7 $\mu$m)', subplot_label='c)', scaling_factor=1e-2)
        self.plot_subplot(axs[1, 0], dataset, 'vil', frame_idx, cmap='Spectral_r', colorbar_label='Vertically Integrated Liquid, [$kg \ m^{-2}$]', title='NEXRAD', subplot_label='d)', scaling_factor=1.0)
        self.plot_subplot(axs[1, 1], dataset, 'lightning_flashes', frame_idx, cmap='magma', colorbar_label='Number of flashes', title='GLM Lightning', subplot_label='e)', scaling_factor=1.0)
        #dont need the 6th subplot, so turn that off
        self.plot_subplot(axs[1, 2], axs[-1].axis('off'))

        # Adjust layout for better appearance
        plt.tight_layout()
        plt.show()