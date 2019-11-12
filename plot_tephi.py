"""
Created on Tue Nov 12 15:39:30 2019

@author: nms6
"""

import tephi
import matplotlib.pyplot as plt

def tephi_plot(input_dict, plot_fname, style_dict=None):
    """
    Plot T-ϕ-gram using modified tephi scripts.
    
    input_dict:
        keys:   data type
        values: 2D numpy array of [pressure level, temperature] 
    style_dict:
        keys:   data type
        values: dictionary {'c':, 'ls':}
    """
    
    fig,ax = plt.subplots(figsize=(10,20))
    plt.axis('off')
    
    # Input tephigram modifications here
    tpg = tephi.Tephigram(figure=fig, anchor=[(1000, -10), (100, -50)])
    tephi.MIN_THETA = -10
    tephi.WET_ADIABAT_SPEC = [(5, None)]
    tephi.MAX_WET_ADIABAT = 60
    tephi.MIXING_RATIO_LINE.update({'linestyle': '--'})
    
    for key,data in input_dict.items():
        
        if style_dict is None:
            tpg.plot(data, label=key)
        else:
            tpg.plot(data, color=style_dict[key]['c'], linestyle=style_dict[key]['ls'], label=key)

    plt.savefig(plot_fname, bbox_inches='tight')

