import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from IPython.display import Image
import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('ReviewsHotel.csv', encoding='latin-1')

TextData = pd.read_csv('PreProcessedData.csv')

f = feature_extraction.text.CountVectorizer()
X = f.fit_transform(TextData["Text"])
X = X.toarray()
print(np.shape(X))

X_train, X_test, y_train, y_test = model_selection.train_test_split(X[0:1600, ], data.Deceptive[0:1600], test_size=0.33,
                                                                    random_state=42)
bayes = naive_bayes.MultinomialNB()
bayes.fit(X_train, y_train)
global score_test_bayes_pre
score_test_bayes_pre = bayes.score(X[0:1600, ], data.Deceptive[0:1600])
global recall_test_bayes_pre
recall_test_bayes_pre = metrics.recall_score(data.Deceptive[0:1600], bayes.predict(X[0:1600, ]))
global precision_test_bayes_pre
precision_test_bayes_pre = metrics.precision_score(data.Deceptive[0:1600], bayes.predict(X[0:1600, ]))
print(score_test_bayes_pre)

rating = np.array(data.Rating[0:1600])
print(data.Rating[0:1600])

X = np.column_stack((X, rating))

X_train, X_test, y_train, y_test = model_selection.train_test_split(X[0:1600, ], data.Deceptive[0:1600], test_size=0.33,
                                                                    random_state=42)
bayes = naive_bayes.MultinomialNB()
bayes.fit(X_train, y_train)
score_test_bayes_pre = bayes.score(X[0:1600, ], data.Deceptive[0:1600])
recall_test_bayes_pre = metrics.recall_score(data.Deceptive[0:1600], bayes.predict(X[0:1600, ]))
precision_test_bayes_pre = metrics.precision_score(data.Deceptive[0:1600], bayes.predict(X[0:1600, ]))

print(score_test_bayes_pre)
print(bayes.predict([X[-1]]))
print(X[-1])

