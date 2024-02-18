import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

class classificator:

    def __init__(self, ):
        clf = DecisionTreeClassifier()


    #temporary method for read data
    def read_data_spam(self, data_name: str = 'spam.csv') -> tuple[pd.DataFrame, pd.DataFrame]:
        data = pd.read_csv('spam.csv')
        return data.Message, data.spamORham
    
    def fit(self, ):
        X, y = self.read_data_spam()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) #test_size must be const or var
        
        return X_train, X_test, y_train, y_test

    def predict(self, ) -> float:
        raise NotImplementedError()

    def score(self, ) -> float:
        raise NotImplementedError()
    
    def get_picture(self, ):
        raise NotImplementedError()






if __name__ == '__main__':
    clf = classificator()

    print(clf)
    print(clf.read_data_spam())
    print(clf.fit())
