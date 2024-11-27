''' *******************************************************************************
Importing Libraries
********************************************************************************'''
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

''' *******************************************************************************
Custom Functions
********************************************************************************'''

# ---------------------------PCAvariance Function ---------------------------------

def PCAvariance(VarRat, VarRatAcum):

    '''
    Description: 
        Function to plot pca variances
    
    Inputs: 
       VarRat -> Ratio of variances related to each pca component
       VarRatCum -> Comulative varaice for the pca components

    '''
    
    fig, ax1 = plt.subplots(figsize=(15,5))
    plt.rcParams.update({'font.size': 14})

    # Plotting the components variances ratios
    ax1.bar(range(len(VarRat)),VarRat, color='c')
    ax1.set_xlabel('PCA components')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Variance ratio', color='k')
    ax1.tick_params('y', colors='k')
    plt.grid(axis='x')

    # Plotting the cumulative variance
    ax2 = ax1.twinx()
    ax2.plot(VarRatAcum, 'r-*')
    ax2.set_ylabel('Commulative Variance ratio', color='r')
    ax2.tick_params('y', colors='r')
    plt.title('Variance accounted by each principal component')
    fig.tight_layout()
    plt.grid(axis='y')
    plt.show()

# ---------------------------PCAcompWeights Function ------------------------------

def PCAcompWeights(pca, df, component, plot=False, figsize=(10,5)):

    '''
    Description: 
        Function to plot variables weights for the selected pca component
    
    Inputs: 
       pca -> pca model containing the pca components
       df -> pandas dataframe contained the data used to train the pca model
       component -> Selected component to explain its composition
       plot -> Variable used to determine if the function displays as result a Figure or a table
       figsize -> Size of the Figure to display, (length, height)
       
    '''
    
    #Creating a dataframe with the pca components
    componentsA = pd.DataFrame(np.round(pca.components_, 4), columns = df.columns)
    
    #Generating a copy of the created dataframe
    components = componentsA.copy()
    
    #Sorting the weigths for the required pca component
    components.sort_values(by=component-1, ascending=False, axis=1, inplace=True)

    #Printing the sorted pca weights for the required component
    if plot==False:
        print(components.loc[component-1].transpose())
    
    #Plotting the sorted pca weights for the required component 
    if plot==True:
        my_colors = ['darkorange', 'royalblue', 'coral', 'springgreen', 'magenta', 'cornflowerblue', 'seagreen']
        plt.rcParams.update({'font.size': 10})
        fig, ax = plt.subplots(figsize = figsize)
        components.loc[component-1].plot(ax = ax, kind = 'bar',  color=my_colors);
        ax.set_ylabel('Weights')
        plt.title('PCA Component Composition - Weights of variables')
        plt.grid()

''' *******************************************************************************
FIN
********************************************************************************'''