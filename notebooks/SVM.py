from sklearn.svm import LinearSVC
import metodos as m

def ObtenerHiperParametros(x_train, y_train, x_test, y_test):
    parametros = [
        { 'penalty': ['l1'], 'C': [1, 10, 100], 'dual': [False], 'class_weight': [{1:4}, {1:5}] }
    ]
    medida = 'recall'

    return m.GridSearch(LinearSVC(), parametros, medida, x_train, y_train, x_test, y_test)

def Entrenar(x_train, y_train, parametros):
    # Después de aplicar Grid Search de determinaron los siguiente parámetros
    # {'C': 1, 'class_weight': {1: 5}, 'dual': False, 'penalty': 'l1'}

    modelo = LinearSVC(C=1, penalty='l1', dual=False, class_weight={1:5})
    modelo.fit(x_train, y_train)
    return modelo