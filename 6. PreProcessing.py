import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import nltk
from nltk.corpus import stopwords

data = pd.read_csv('ReviewsHotel.csv', encoding='latin-1')
nltk.download('stopwords')
stop = set(stopwords.words('english'))
print("These are the list of stop words")
print(stop)

final_X = data['Text']

print("These are the reviews")
print(final_X)

import re

temp = []
snow = nltk.stem.SnowballStemmer('english')
for sentence in final_X:
    sentence = sentence.lower()  # Converting to lowercase
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', sentence)  # Removing HTML tags
    sentence = re.sub(r'[?|!|\'|"|#|:|;]', r'', sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]', r' ', sentence)  # Removing Punctuations

    words = [snow.stem(word) for word in sentence.split() if
             word not in stopwords.words('english')]  # Stemming and removing stopwords
    temp.append(words)

final_X = temp

print(final_X[0:10])

sent = []
for row in final_X:
    sequ = ''
    for word in row:
        sequ = sequ + ' ' + word
    sent.append(sequ)

final_X = sent

final_X = list(final_X)

print("After Pre-processing")
print(final_X[0:5])
print(type(final_X))

f = open('PreProcessedData.csv', 'w')
f.write("Text" + '\n')
for ele in final_X:
    f.write(ele + '\n')
f.close()
