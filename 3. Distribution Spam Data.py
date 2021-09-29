import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from IPython.display import Image
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv('ReviewsHotel.csv', encoding='latin-1')

count_Class = pd.value_counts(data["Deceptive"], sort=True)
count_Class.plot(kind='bar', color=["blue", "orange"])
plt.title('Bar chart')
plt.show()

count_Class.plot(kind = 'pie',  autopct='%1.0f%%')
plt.title('Pie chart')
plt.ylabel('')
plt.show()
