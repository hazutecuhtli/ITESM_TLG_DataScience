import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from IPython.display import display_html
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor



def plottin_trend(df, col1, col2, title,x_scl = 1.5, y_scl=1.5):
    
    fig = plt.figure(figsize=(10,4), facecolor='lightgray')

    fig.subplots_adjust(left=0.2,wspace = 0.1)
    ax1 = plt.subplot2grid((1,2),(0,0)) 
    bbox=[0, 0, 1, 1]
    ax1.axis('off')

    df_tmp = df.groupby(col1)[[col2]].agg(['count', 'min', 'max', 'mean']).reset_index()
    #df_tmp.set_index(col1, inplace=True)
    df_tmp.index.name = ''
    df_tmp.columns = [col[1] for col in df_tmp.columns]
    df_tmp = df_tmp.round()
    df_tmp = df_tmp
    

    table = ax1.table(cellText=df_tmp.values, bbox=[0,0,1,.7], colLabels=df_tmp.columns, colWidths=[.15,.1,.1,.1,.1])
    table.set_fontsize(14)
    table.scale(x_scl, y_scl)
    ax1.set_title(title + '- Summary')
    
    for (row,col), cell in table.get_celld().items():

        if row == 0:
            cell.set_text_props(weight='bold', horizontalalignment='right')
        if (row == 0) & (col==0):
            cell.set_facecolor('white')
            cell.set_edgecolor('white')    
        elif (row == 0) & (col==1):
            cell.set_text_props(weight='bold', horizontalalignment='right')
            cell.set_facecolor('white')
            cell.set_edgecolor('white')
        
        elif (row%2 == 0):
            if  col == 0:
                cell.set_text_props(weight='bold', horizontalalignment='center', fontsize=12)
            cell.set_edgecolor('white')
            cell.set_facecolor('white')
        elif (row%2 == 1):
            if col == 0:
                cell.set_text_props(weight='bold', horizontalalignment='center')
            cell.set_edgecolor('white')
            cell.set_facecolor('whitesmoke')        
        
        

        
    #ax2 = axs[1]
    ax2 = plt.subplot2grid((1,2),(0,1),colspan=1) 
    plt.subplots_adjust(wspace=0.0,hspace=0.1)
    ax2.set_title(title + '- Boxplot', fontsize=12)
    colors = ['mediumaquamarine', 'coral', 'cornflowerblue', 'violet', 'yellowgreen', 'gold']
    ax2 = sns.boxplot(df, x=col, y=col2, hue=col1,width=0.8, gap=.3);
    #ax2.set_xlim([-.15,.15])
    plt.tight_layout()




def Removing_IQROutliers(df, cols):
    # Looping through the Variables of Interests
    for col in cols:
    
      quartiles = df[[col]].quantile([.25, .50, .75], axis = 0).values.transpose()[0]
      # Estimating the Interquartile Range
      IQR = quartiles[2] - quartiles[0]
      # Degining the Quartiles Fences
      lower_fence = quartiles[0]-1.5*IQR
      upper_fence = quartiles[2]+1.5*IQR
      # Displaying reduction in the information size, the removing of outliers
      print(col, df[cols].count().values[0], IQR, quartiles, lower_fence, upper_fence)
      # Removing Outliers
      return df[(df[col]>=lower_fence) & (df[col]<=upper_fence)]


def display_dfs(dfs, gap=50, justify='center'):
    html = ""
    for title, df in dfs.items():  
        df_html = df._repr_html_()
        cur_html = f'<div> <h3>{title}</h3> {df_html}</div>'
        html +=  cur_html
    html= f"""
    <div style="display:flex; gap:{gap}px; justify-content:{justify};">
        {html}
    </div>
    """
    display_html(html, raw=True)


def graficar(X, Y, stan_res, model_loc, figsize=(8,8)):

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    ########## Normal Probability Plot
    
    (quantiles, values), (slope, intercept, r) = stats.probplot(stan_res, dist='norm')    
    axes[0, 0].plot(values, quantiles, 'ob')    
    axes[0, 0].plot(quantiles * slope + intercept, quantiles, 'r')    
    axes[0, 0].set_title('Normal Probability Plot')    
    axes[0, 0].set_ylabel('Percent')    
    axes[0, 0].set_xlabel('Standardized Residual')    
    ticks_perc=[1,10,50,90,99]    
    ticks_quan=[stats.norm.ppf(i/100.) for i in ticks_perc]    
    axes[0, 0].set_xticks([-2,0,2,4])    
    axes[0, 0].set_yticks(ticks_quan, ticks_perc)    
    axes[0, 0].grid()
    
    
    ########### Versus Fits
    
    model_fitted_y = model_loc.fittedvalues    
    sns.residplot(x=model_fitted_y, y=stan_res, ax=axes[0,1])
    axes[0, 1].set_title('Versus Fits')    
    axes[0, 1].set_xlabel('Fitted Value')    
    axes[0, 1].set_ylabel('Standardized Residual')    
    #axes[0, 1].set_xticks([0,4000,8000,12000,16000])
    
    #axes[0, 1].set_yticks([-2,0,2,4])
    
    
    ########### Histogram
    
    stan_res = model_loc.get_influence().resid_studentized_internal    
    sns.histplot(stan_res, bins=10, ax=axes[1,0])    
    axes[1, 0].set_title('Histogram')
    axes[1, 0].set_ylabel('Frequency')    
    axes[1, 0].set_xlabel('Standardized Residual')    
    axes[1, 0].set_yticks([0,2.5,5,7.5,10])
    
    
    ########### Versus Order
    
    axes[1, 1].scatter(X.index, stan_res)    
    axes[1, 1].plot(X.index,stan_res)    
    axes[1, 1].axhline(y=0,color='grey',linestyle=':')    
    axes[1, 1].set_title('Versus Order')    
    axes[1, 1].set_xlabel('Observation Order')    
    axes[1, 1].set_ylabel('Standardized Residual')
    
    #axes[1, 1].set_xticks(np.arange(0,31,2))
    
    #axes[1, 1].set_yticks([-2,0,2,4]);

    plt.tight_layout()


def Coefficients_Table(df, model, target, Y, Y_pred):

    print('Equation Found: y = ', str(round(model.params[0], 4)) + ' + \n                     ' + '                     '.join([str(round(coef,4)) + ' ' + str(var) + '\n' for var, coef in zip(model.params[1:].index.tolist(), model.params[1:])]))
    
    MSE = mean_squared_error(Y, Y_pred)
    MAE = mean_absolute_error(Y, Y_pred)
    
    metrics = {'DoF':int(model.df_model), 'F-Value':[model.fvalue], 'P-value(F)':[model.f_pvalue], 'R-Squared':[100*model.rsquared],
               'Adj. R-squared':[100*model.rsquared_adj], 'MAE':MAE, 'MSE':MSE}
    
    display(pd.DataFrame(metrics, index=['Scores']))

    vif_data = pd.DataFrame()
    cols = model.params.index[1:].tolist()
    vif_data["feature"] = model.params.index[1:].tolist()
    
    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(df[cols].values, i)
                              for i in range(len(df[cols].columns))]
    
    dict_test = {'Coef':model.params.values, 'SE Coef':model.bse.values, 
                 'Corr. Coef':[np.nan] + df[cols+[target]].corr()[target].tolist()[:-1],
                 'T-value':model.tvalues.values,
                 'P-value(T)':model.pvalues.values, 'VIF':np.array([np.nan]+vif_data['VIF'].tolist())}

    df_table = pd.DataFrame(dict_test, index=model.params.index.tolist())
    
    display(df_table)