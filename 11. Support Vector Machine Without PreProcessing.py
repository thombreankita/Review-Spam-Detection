import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from IPython.display import Image
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('ReviewsHotel.csv', encoding='latin-1')
data=data[0:1600]
TextData = pd.read_csv('PreProcessedData.csv', encoding='latin-1')
TextData=TextData[0:1600]

f = feature_extraction.text.CountVectorizer(stop_words='english')
X = f.fit_transform(TextData["Text"])
X = X.toarray()
rating = np.array(data.Rating)
X = np.column_stack((X, rating))

print("We remove the stop words in order to improve the analytics")
print(np.shape(X))

print("First we transform the variable spam/non-spam into binary variable, then we split our data set in training "
      "set and test set.")

X_train, X_test, y_train, y_test = model_selection.train_test_split(X[0:1600,], data.Deceptive[0:1600], test_size=0.33,
                                                                    random_state=42)
print([np.shape(X_train), np.shape(X_test)])

svc = svm.SVC()
svc.fit(X_train, y_train)
score = svc.score(X, data["Deceptive"])
recall = metrics.recall_score(data["Deceptive"], svc.predict(X))
precision = metrics.precision_score(data["Deceptive"], svc.predict(X))

print("Accuracy:")
print(score)
print("Recall:")
print(recall)
print("Precision:")
print( precision)


performance = [round(score * 100,2), round(recall * 100,2), round(precision * 100,2)]
objects = performance
y_pos = np.arange(len(objects))


plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Percentage')
plt.xlabel('Accuracy          Recall             Precision')
plt.title('Accuracy Parameters')

plt.show()
