import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('image', cmap='copper')

# not used
from matplotlib.colors import ListedColormap
# custom colormap
def mycmap_colors():
    colors = [(0.0, 0.0, 0.0, 1),(1.0, 0.66, 0.0, 1)]
    return ListedColormap(colors)
mycmap = mycmap_colors()


# a nonlinear function of a 2d array x
def NF(x,B,c=1):
    r=0
    if c==3:
        if np.sin((0.5*x[0]+0.5*x[1])*4*np.pi/B) * np.cos((-0.5*x[0]+0.5*x[1])*2*np.pi/B)>0:
            r=1
    if c==4:
        if np.sin((0.5*x[0]+0.5*x[1])*8*np.pi/B) * np.cos((-0.5*x[0]+0.5*x[1])*6*np.pi/B)>0:
            r=1
    return r

def plot_data(x,y):  
    plt.figure(figsize = (6,6))
    plt.scatter(x[:,0],x[:,1],s=6,c=y)
    plt.show()


def filename(s,L,TYPE=1):
    return "./DATA/"+s+"-for-DNN_type"+str(TYPE)+"_L"+str(L)+".dat"
