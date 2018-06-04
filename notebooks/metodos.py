from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from imblearn.over_sampling import ADASYN 
from sklearn.feature_selection import RFECV
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from itertools import product
import numpy as np

def MostrarProporciones(num_tv, num_tf):
    print('Cantidad de transacciónes válidas: {0}'.format(num_tv))
    print('Cantidad de transacciónes fraudulentas: {0}'.format(num_tf))
    print('Grado de desbalanceamiento: {0}'.format(num_tf/num_tv))

def GridSearch(modelo, parametros, medida, x_train, y_train, x_test, y_test):        
    clf = GridSearchCV(modelo, parametros, cv=5, scoring=medida)
    clf.fit(x_train, y_train)

    return clf.best_params_

def ObtenerADASYN(x_train, y_train):
    adasyn = ADASYN(random_state=0)
    return adasyn.fit_sample(x_train, y_train)

def EntrenarConFRyCV(modelo, x_train, y_train):
    rfecv = RFECV(estimator=modelo, step=1, cv=5, scoring='recall')
    rfecv.fit(x_train, y_train)

def MostrarMatrizDeConfusion(modelo, x_test, y_test):
    from sklearn import metrics

    predicciones_test = modelo.predict(x_test)
    matriz_confusion = metrics.confusion_matrix(y_test, predicciones_test)

    TN = matriz_confusion[0,0]
    FN = matriz_confusion[1,0]
    FP = matriz_confusion[0,1]
    TP = matriz_confusion[1,1]

    print ('              +-----------------+')
    print ('              |   Predicción    |')
    print ('              +-----------------+')
    print ('              |    +   |    -   |')
    print ('+-------+-----+--------+--------+')
    print ('| Valor |  +  |   %d |   %d   |'   % (TP, FN) )
    print ('| real  +-----+--------+--------+')
    print ('|       |  -  |   %d  |   %d  |'    % (FP, TN) )
    print ('+-------+-----+--------+--------+')
    print ()
    print ( 'Exactitud        : ', (TP+TN)/(TP+FN+FP+TN) )
    print ( 'Precición        : ', (TP)/(TP+FP) )
    print ( 'Exhaustividad    : ', (TP)/(TP+FN) )

def ObtenerMedidaDeCalidad(modelo, x, y):
    y_score = modelo.decision_function(x)
    return average_precision_score(y, y)

def MostrarMedidaDeCalidad(modelo, x, y):
    y_score = modelo.decision_function(x)
    average_precision = average_precision_score(y, y_score)
    print('Medida del promedio de la Precisión y Exhaustividad: {0:0.2f}'.format(average_precision))

    precision, recall, _ = precision_recall_curve(y, y_score)

    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Exhaustividad')
    plt.ylabel('Precisión')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Curva de Precisión y Exhaustividad: AP={0:0.2f}'.format(average_precision))

def MostrarRegionesDeDecision(modelos, nombres, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))

    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

    for idx, clf, tt in zip(product([0, 1], [0, 1]), modelos, nombres):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
        axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        axarr[idx[0], idx[1]].set_title(tt)

    plt.show()