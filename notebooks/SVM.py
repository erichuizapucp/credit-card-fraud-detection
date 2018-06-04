from sklearn.svm import LinearSVC
import metodos as m

def ObtenerHiperParametros(x_train, y_train, x_test, y_test):
    parametros = [
        { 'penalty': ['l1'], 'C': [1, 10, 100], 'dual': [False], 'class_weight': [{1:4}, {1:5}] }
    ]
    medida = 'recall'

    return m.GridSearch(LinearSVC(), parametros, medida, x_train, y_train, x_test, y_test)

def Entrenar(x_train, y_train, parametros):
    modelo = LinearSVC(C=parametros['C'], penalty=parametros['penalty'], dual=parametros['dual'], class_weight=parametros['class_weight'])
    modelo.fit(x_train, y_train)
    return modelo