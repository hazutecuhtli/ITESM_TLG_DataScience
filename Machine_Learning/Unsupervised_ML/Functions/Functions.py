''' *******************************************************************************
Importing Libraries
********************************************************************************'''
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn import cluster, metrics
from sklearn.cluster import KMeans

''' *******************************************************************************
Custom Functions
********************************************************************************'''


def PCATest(df, n_scor, rango, layout=(1,1), Figsize=(15,13)):

    fig, ax = plt.subplots(layout[0], layout[1], figsize=Figsize)
    
    for i, m in enumerate(n_scor):

        df = df[0:rango, :]

        interia = []

        for center in range(1,m+1):
    
            kmeans = KMeans(center, random_state=0, n_init='auto')
    
            model = kmeans.fit(df)

            interia.append(model.inertia_)
    
        centers = list(range(1,m+1))

        plt.subplot(layout[0], layout[1], i+1) 
        plt.plot(centers,interia)
        plt.title('Kmeans (PCA components)')
        plt.xlabel('Centers');
        plt.ylabel('Average distance');
        plt.grid()
        i+=1
        
    plt.subplots_adjust(top=1, bottom=0.08, left=0.10, right=0.95, hspace=0.3, wspace=0.35);
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

# ---------------- Plotting_SilhouetteCoef_TLG Function ------------------------------

def Plotting_SilhouetteCoef_TLG(df, nc):


    rows = int(np.ceil((nc-2)/4))
    colums = 4 #round(df_encoded[cols].values.shape[1]/4)
    
    fig, axs = plt.subplots(rows, colums, figsize=(11,2+rows*3), facecolor='lightgray', sharex=False, sharey=False)
    fig.suptitle('Silhouette analysis for KMeans clustering on sample data')
    count = 1
    for k in range(2, nc):
        ax = plt.subplot(rows, colums, count)
        ax.set_ylim([0, df.shape[0] + (k + 1) * 10])
        k_means = KMeans(n_clusters=k, random_state=0, n_init='auto')
        y_pred = k_means.fit_predict(df)
        silhouette_avg = metrics.silhouette_score(df, y_pred)
        sample_silhouette_values = metrics.silhouette_samples(df, y_pred)
    
        y_lower = 10
        
        for i in range(k):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
            ith_cluster_silhouette_values = \
            sample_silhouette_values[y_pred == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / k)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
            ax.axvline(x=silhouette_avg, color="red", linestyle="--")
            if (count%1)==0:
                ax.text(-0.02, y_lower + 0.1 * size_cluster_i, str(i))

            elif (count%2)==0:
                ax.text(-0.02, y_lower + 0.1 * size_cluster_i, str(i))
            elif (count%3)==0:
                ax.text(-0.02, y_lower + 0.1 * size_cluster_i, str(i))
            else:
                ax.text(-0.02, y_lower + 0.1 * size_cluster_i, str(i))
           
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
      
        count += 1
        #ax.set_xticklabels([], fontsize=12)
        ax.set_title("Coefficients for {} clusters".format(k))
        ax.set_xlabel("Coefficients")
        ax.set_ylabel("Cluster")
        ax.set_yticks([])  # Clear the yaxis labels / ticks
    
    plt.tight_layout()

# ---------- Plotting_Predictions_DfComponents Function ------------------------------


def Plotting_Predictions_DfComponents(df, predict, clusters, col_base):

    rows = int(np.ceil(df.shape[1]/4))
    colums = 4 #round(df_encoded[cols].values.shape[1]/4)
    
    fig, axs = plt.subplots(rows, colums, figsize=(11,2+rows*2.5), facecolor='lightgray', sharex=False, sharey=False)
    fig.suptitle('Silhouette analysis for KMeans clustering on sample data')
    count = 1
    
    for k, col in enumerate(df.columns):

        ax = plt.subplot(rows, colums, count)
        colors = cm.nipy_spectral(predict.astype(float)/clusters)
        ax.scatter(df[col_base].values, df[col].values, c=colors)
        #ax1.set_ylim(0, 0.5e7)
        ax.set_title("The visualization of the clustered data.")
        ax.set_xlabel('test')
        ax.set_ylabel('price')
        
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                        "with n_clusters = %d" % k),
                        fontsize=14, fontweight='bold')
      
        count += 1
        #ax.set_xticklabels([], fontsize=12)
        ax.set_title("Data - Segmentation")
        ax.set_xlabel(col_base)
        ax.set_ylabel(col)
        ax.set_yticks([])  # Clear the yaxis labels / ticks
    
    plt.tight_layout()


''' *******************************************************************************
FIN
********************************************************************************'''