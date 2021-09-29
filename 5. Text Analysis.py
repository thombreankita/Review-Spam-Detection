import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from IPython.display import Image
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv('ReviewsHotel.csv', encoding='latin-1')

count1 = Counter(" ".join(data[data['Deceptive'] == 1]["Text"]).split()).most_common(35)
df1 = pd.DataFrame.from_dict(count1)
df1 = df1.rename(columns={0: "words in non-spam", 1: "count"})
count2 = Counter(" ".join(data[data['Deceptive'] == 0]["Text"]).split()).most_common(35)
df2 = pd.DataFrame.from_dict(count2)
df2 = df2.rename(columns={0: "words in spam", 1: "count_"})

df1.plot.bar(legend=False)
y_pos = np.arange(len(df1["words in non-spam"]))
plt.xticks(y_pos, df1["words in non-spam"])
plt.title('More frequent words in Deceptive Hotel Reviews')
plt.xlabel('words')
plt.ylabel('number')
plt.show()

df2.plot.bar(legend=False, color='orange')
y_pos = np.arange(len(df2["words in spam"]))
plt.xticks(y_pos, df2["words in spam"])
plt.title('More frequent words in Truthful Hotel Reviews')
plt.xlabel('words')
plt.ylabel('number')
plt.show()
