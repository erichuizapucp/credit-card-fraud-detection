from sklearn.naive_bayes import GaussianNB
import metodos as m

def Entrenar(x_train, y_train):

    modelo = GaussianNB()
    modelo.fit(x_train, y_train)
    
    return modelo