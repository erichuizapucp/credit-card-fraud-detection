from sklearn.neighbors import KNeighborsClassifier

def Entrenar(x_train, y_train):
    modelo = KNeighborsClassifier(n_neighbors=2, p=1, weights='distance', metric='manhattan')
    modelo.fit(x_train, y_train)
    return modelo