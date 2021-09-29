import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from IPython.display import Image
import warnings


warnings.filterwarnings("ignore")
data = pd.read_csv('ReviewsHotel.csv', encoding='latin-1')

f = feature_extraction.text.CountVectorizer()
X = f.fit_transform(data["Text"])

print(np.shape(X))

print("First we transform the variable spam/non-spam into binary variable, then we split our data set in training "
      "set and test set.")

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['Deceptive'], test_size=0.33,
                                                                    random_state=42)
print([np.shape(X_train), np.shape(X_test)])

bayes = naive_bayes.MultinomialNB()
bayes.fit(X_train, y_train)
score = bayes.score(X, data["Deceptive"])
recall = metrics.recall_score(data["Deceptive"], bayes.predict(X))
precision = metrics.precision_score(data["Deceptive"], bayes.predict(X))

print("Accuracy:")
print(score)
print("Recall:")
print(recall)
print("Precision:")
print(precision)


performance = [round(score * 100,2), round(recall * 100,2), round(precision * 100,2)]
objects = performance
y_pos = np.arange(len(objects))


plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Percentage')
plt.xlabel('Accuracy          Recall             Precision')
plt.title('Accuracy Parameters')

plt.show()
