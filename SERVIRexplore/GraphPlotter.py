import matplotlib
import numpy as np

class GraphPlotter:
    def __init__(self):
        pass  # You may not need an __init__ method for this example

    def plot_graph(self, groups, datasrc):

        #for plotting purposes I want to reverse that 
        groups = list(groups)
        groups = groups[::-1]

        ###### plot stuff #########
        bins=np.arange(-100,0,4)
        #midpoints of the bins
        mids=np.arange(-98,-2,4)

        #colormap to assign a color for each month 
        cmap = matplotlib.cm.Spectral_r
        bounds = np.arange(0,1.3,.1)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        scalarMap = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        ###########################

        ###### Actual Plot #########
        #make fig
        fig = matplotlib.pyplot.figure(figsize=(5,7.5))
        #set not clear facecolor
        fig.set_facecolor('w')
        #grab axis handle 
        ax = matplotlib.pyplot.gca()
        #loop over all groups 
        for idx,group in groups:
            #get the counts per bin 
            c,b = np.histogram(group.values,bins=bins)
            #normalize 
            c = c/c.sum()
            #plot axis line
            ax.axhline((idx/10),color='k',zorder=0,alpha=0.5)
            #plot curve
            ax.fill_between(mids,(idx/10),c+(idx/10),alpha=0.9,facecolor=scalarMap.to_rgba((idx/10)),edgecolor='k')

        #set plot xlimits
        ax.set_xlim([-90,-10])
        #force ticks to show up for each group
        ax.set_yticks(np.arange(0,1.3,.1))
        #force ylimits
        ax.set_ylim([0.05,1.4])
        #force yticklabels
        ax.set_yticklabels(['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        #add axis labels
        if datasrc == "IR" : 
            ax.set_xlabel('IR Minimum Brightness Temperature, [$\degree$C]')
        elif datasrc == "VIL": 
            ax.set_xlabel('IVertically Integrated Liquid')
        elif datasrc == "VIS": 
            ax.set_xlabel('Visible Reflectance')

        ax.set_ylabel('Grouped Month')

        print("Inside Graph plotter")

        matplotlib.pyplot.tight_layout()