from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def Entrenar(x_train, y_train):
    clasificadorRandomForest = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
    clasificadorRandomForest.fit(x_train, y_train)

    return clasificadorRandomForest