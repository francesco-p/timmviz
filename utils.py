import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import torchshow as ts

def process_attrs(var, attrs):
    """ Split the attributes by . into a list
    and apply getattr to each attribute. If the attr is a digit
    then convert it to an int and apply getattr to the last attribute.

    Args:
        attrs (list): list of attributes
    
    Returns:
        list: list of attributes
    """
    apply_getattr = lambda var, attr: getattr(var, attr) if not attr.isdigit() else var[int(attr)]
    
    for attr in attrs:
        var = apply_getattr(var, attr)
    
    return var


# def show_hist(x, bins):
#     fig, ax = plt.subplots()
#     y = torch.histc(x, bins=bins, min=x.min().item(), max=x.max().item())
#     ax.bar(range(bins), y, align='center', alpha=0.5)
#     # Set xticks to min and max of x
#     xticklabels = ['']*bins
#     xticklabels[0] = x.min().item()
#     xticklabels[-1] = x.max().item()
#     ax.set_xticklabels(xticklabels)
#     return fig, ax


def model_size(model):
    """ Prints model info """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mb = (params * 32) / 2**23
    gb = (params * 32) / 2**33
    #return params, mb, gb
    dictionary = {'Params': [params], 'MB': [f'{mb:.2f}'], 'GB': [f'{gb:.2f}']}
    return pd.DataFrame(dictionary)


def show_hist(x, nbins):
    # Convert PyTorch tensor to numpy array
    x = x.numpy().flatten()
    
    # Find most common value
    mean = x.mean()
    
    # Create figure and axis using matplotlib
    fig, ax = plt.subplots()
    
    # Plot histogram
    ax.hist(x, bins=nbins)
    
    # Set axis labels and title
    ax.set_xlabel(f'Values (bins={nbins})')
    ax.set_ylabel('Counts')
    #ax.set_title('Histogram of Module Params')
    
    # Show minimum, maximum, and most common value on x-axis
    ax.set_xticks([x.min(), x.max(), mean])
    ax.set_xticklabels([f'{x.min():.3f}\nmin', f'{x.max():.3f}\nmax', f'{mean:.3f}\n$\mu$'])
    plt.tight_layout()
    plt.savefig('imgs/hist.png')
    plt.close()
    return




