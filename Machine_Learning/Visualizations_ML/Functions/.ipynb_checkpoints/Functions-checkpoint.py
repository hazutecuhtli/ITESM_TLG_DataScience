from yellowbrick.classifier.threshold import DiscriminationThreshold
from yellowbrick.classifier import (
    ConfusionMatrix, ClassPredictionError, ClassificationReport,
    PrecisionRecallCurve, ROCAUC, ClassPredictionError
)
from yellowbrick.model_selection import FeatureImportances
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def fun_plot_grid_search_results(df):   
    '''Regresa un DataFrame con los resultados de la exploración de hiperparámetros.

    Esta función muestra una tabla estilizada con los resultados de la exploración de
    hiper parámetros utilizando validación cruzada.
    
    En la tabla aparecen del lado izquierdo la combinación de hiperparámetros, el error
    de entrenamiento y el error de validación cruzada y el ranking del mejor modelo
    a partir del error de validación.
    
    El mapa de color va de azul (modelos con menor error) a rojo (modelos con mayor error).
    Se calcula para el error de entrenamiento y validación de manera separada.
    
    Los resultados están ordenados de mejor a peor modelo, en función del error de validación.
    
    Parameters
    ----------
    dict: Diccionario con los resultados de la búsqueda de hiperparámetros GridSearchCV.cv_results_ 
        
    Returns
    -------
    pandas.io.formats.style.Styler
        Regresa una tabla estilizada con los resultados de la búsqueda de hiperparámetros.
    '''
    # Elegir paleta divergente de colores
    cm = sns.diverging_palette(5, 250, as_cmap=True)
    
    return (
        pd.concat([
            # Limpiar la columna de parámetros
            df['params'].apply(pd.Series), 
            # Extraer solamente el error de prueba 
            df[['mean_train_score', 'mean_test_score', 'rank_test_score']]],
            axis = 1
        )
        # Ordenar los modelos de mejor a peor
        .sort_values(by = 'rank_test_score')
        # Pintar el fondo de la celda a partir del error de validación
        .style.background_gradient(cmap=cm, subset = ['mean_train_score', 'mean_test_score'])
    )
    
def fun_graficar_matriz_confusion(model, X_val_test, y_val_test):
    '''Graficar matriz de confusión
    
    Parameters
    ----------
    model (Estimator): Fitted estimator
    X_test (DataFrame): Predictores del conjunto de validación o prueba.
    y_test (ndarray): Clases del conjunto de validación o prueba.
    ''' 
    fig, ax = plt.subplots(figsize = (5,5))
    cm = ConfusionMatrix(
        model,
        classes=['no', 'yes']
    )
    cm.fit(X_train, y_train)
    cm.score(X_val_test, y_val_test)
    cm.show();
    
def fun_graficar_reporte_clasificacion(model, X_val_test, y_val_test):
    '''Graficar reporte de clasificación
    
    Parameters
    ----------
    model (Estimator): Fitted estimator
    X_test (DataFrame): Predictores del conjunto de validación o prueba.
    y_test (ndarray): Clases del conjunto de validación o prueba.
    '''   
    fig, ax = plt.subplots(figsize = (6,5))
    visualizer = ClassificationReport(
        model,
        classes=['no', 'yes']
    )
    visualizer.fit(X_train, y_train)
    visualizer.score(X_val_test, y_val_test)
    visualizer.show();  
    
def fun_graficar_curva_precision_recall(model, X_val_test, y_val_test):
    '''Graficar curva de precision y recall
    
    Parameters
    ----------
    model (Estimator): Fitted estimator
    X_test (DataFrame): Predictores del conjunto de validación o prueba.
    y_test (ndarray): Clases del conjunto de validación o prueba.
    '''
    fig, ax = plt.subplots(figsize = (6,6))
    viz = PrecisionRecallCurve(model)
    viz.fit(X_train, y_train)
    viz.score(X_val_test, y_val_test)
    viz.show();
    
def fun_graficar_curva_roc(model, X_val_test, y_val_test):
    '''Graficar curva ROC
    
    Parameters
    ----------
    model (Estimator): Fitted estimator
    X_test (DataFrame): Predictores del conjunto de validación o prueba.
    y_test (ndarray): Clases del conjunto de validación o prueba.
    '''
    fig, ax = plt.subplots(figsize = (6,6))
    roc_curves_visualizer = ROCAUC(
        model,
        classes=['no', 'yes']
    )
    roc_curves_visualizer.fit(X_train, y_train)
    roc_curves_visualizer.score(X_val, y_val) 
    roc_curves_visualizer.show();
    
def fun_imprimir_roc_auc_score(model, X_val_test, y_val_test):
    '''Imprime el ROC AUC score
    
    Parameters
    ----------
    model (Estimator): Fitted estimator de un DecisionTree, Random Forest o xgboost.
    X_val_test (DataFrame): Clases actuales
    y_val_test (ndarray): Probabildades de clase
    '''
    print(
        "roc_auc_score: {}".format(
            np.round(roc_auc_score(y_val_test, model.predict_proba(X_val_test)[:,1]), 3)
        )
    )
    
def fun_graficar_error_clasificacion(model, X_val_test, y_val_test):
    '''Graficar error de clasificación
    
    Parameters
    ----------
    model (Estimator): Fitted estimator
    X_test (DataFrame): Predictores del conjunto de validación o prueba.
    y_test (ndarray): Clases del conjunto de validación o prueba.
    '''
    fig, ax = plt.subplots(figsize = (7,6))
    cpe_viz = ClassPredictionError(model, classes = ['no', 'yes'])
    cpe_viz.fit(X_train, y_train)
    cpe_viz.score(X_val_test, y_val_test)
    cpe_viz.show();

def fun_graficar_importancias(model):
    '''Graficar importancia de características
    
    Parameters
    ----------
    model (Estimator): Fitted estimator de un DecisionTree, Random Forest o xgboost.
    '''
    fig, ax = plt.subplots(figsize = (8,6))
    viz = FeatureImportances(model)
    viz.fit(X_train, y_train)
    viz.show();
    
def fun_graficar_discrimination_threshold(model, X_val_test, y_val_test):
    '''Graficar relación de precision y recall con distintos puntos de corte
    
    Parameters
    ----------
    model (Estimator): Fitted estimator de un DecisionTree, Random Forest o xgboost.
    X_val_test (DataFrame): Clases actuales
    y_val_test (ndarray): Probabildades de clase
    '''
    fig, ax = plt.subplots(figsize = (8,6))
    visualizer = DiscriminationThreshold(
        model,
        exclude = ["queue_rate"]
    )
    visualizer.fit(X_val_test, y_val_test)
    visualizer.show();     