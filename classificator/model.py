import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay

class classificator:

    def __init__(self, ):
        self.clf = DecisionTreeClassifier()
        self.vectorizer = TfidfVectorizer()
        self.X_train, self.X_test, self.y_train, self.y_test = [None for _ in range(4)]
        self.y_pred = None

    #temporary method for read data
    def read_data_spam(self, data_name: str = 'spam.csv') -> tuple[pd.DataFrame, pd.DataFrame]:
        data = pd.read_csv('spam.csv')
        return data.Message, data.spamORham
    
    def fit(self, ): #EDIT one of parameter must be criterion and max_depth and exclude data parsing from method
        X, y = self.read_data_spam()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3) #test_size must be const or var
        self.X_train, self.X_test = self.to_vectorize(self.X_train, self.X_test)
        self.clf.fit(self.X_train, self.y_train)
        
    def predict(self, ) -> float:
        self.y_pred = self.clf.predict(self.X_test) 
        return self.y_pred

    def score(self, ) -> tuple[float, float]:
        return self.clf.score(self.X_train, self.y_train), self.clf.score(self.X_test, self.y_test)
    
    def get_picture(self, ):
        ConfusionMatrixDisplay.from_predictions(self.y_test, self.y_pred)
        plt.savefig("ConfMatrix")
    
    def to_vectorize(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.vectorizer.fit_transform(X_train), self.vectorizer.transform(X_test)

'''
if __name__ == '__main__':
    clf = classificator()

    print(clf)
    #print(clf.read_data_spam())
    print(clf.fit())
    print(clf.predict())
    print(clf.score())
    clf.get_picture()'''
