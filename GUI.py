import PIL as pl
import matplotlib
from PIL import Image, ImageTk
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from csv import writer
import csv

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from tkinter import *
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm

data = pd.read_csv('ReviewsHotel.csv', encoding='latin-1')
data = data[0:1600]
TextData = pd.read_csv('PreProcessedData.csv', encoding='latin-1')
TextData = TextData[0:1600]

LARGE_FONT = ("Cambria", 16)
SMALL_FONT = ("Cambria", 14)

Directory = "C:/Users/thomb/PycharmProjects/Final"


class SpamDetectionApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.iconbitmap(self)
        tk.Tk.wm_title(self, "Review Spam Detection Using Machine Learning Algorithm")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo, PageThree, PageFour, PageFive, PageSix, PageSeven, PageEight, PageNine,
                  PageTen, PageEleven, PageTwelve, PageThirteen):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, background="snow")
        titleLabel = tk.Label(self, text="Review Spam Detection Using Machine Learning Algorithm", font=("Cambria", 20),
                              background="snow", fg="deep sky blue")
        titleLabel.pack(pady=10, padx=10)

        label = tk.Label(self, text="Home", font=LARGE_FONT, background="snow", fg="light sky blue")
        label.pack(pady=10, padx=10)

        aboutFrame = tk.LabelFrame(self, text="Know About Your Data", font=LARGE_FONT, background="snow",
                                   fg="light sky blue", height=15, width=15)
        aboutFrame.pack(pady=10, padx=10, side=LEFT)

        button = ttk.Button(aboutFrame, text="World Cloud ",
                            command=lambda: controller.show_frame(PageOne))
        button.pack(pady=10, padx=10)

        button2 = ttk.Button(aboutFrame, text="Word Count Graph",
                             command=lambda: controller.show_frame(PageTwo))
        button2.pack(pady=10, padx=10)

        button3 = ttk.Button(aboutFrame, text="Data Distribution",
                             command=lambda: controller.show_frame(PageThree))
        button3.pack(pady=10, padx=10)

        preProcessingFrame = tk.LabelFrame(self, text="Pre Processing of Data", font=LARGE_FONT, background="snow",
                                           fg="light sky blue", height=40, width=40)
        preProcessingFrame.pack(pady=10, padx=10, side=LEFT)

        button4 = ttk.Button(preProcessingFrame, text="Word Count Graph after Pre-processing",
                             command=lambda: controller.show_frame(PageFour))
        button4.pack(pady=10, padx=10)

        button5 = ttk.Button(preProcessingFrame, text="WordCloud after Pre-processing",
                             command=lambda: controller.show_frame(PageFive))
        button5.pack(pady=10, padx=10)

        MachineLFrame = tk.LabelFrame(self, text="Machine Learning Algorithms", font=LARGE_FONT, background="snow",
                                      fg="light sky blue")
        MachineLFrame.pack(pady=10, padx=10, side=LEFT)

        button6 = ttk.Button(MachineLFrame, text="Naive Bayes",
                             command=lambda: controller.show_frame(PageSix))
        button6.pack(pady=10, padx=10)

        button7 = ttk.Button(MachineLFrame, text="Support Vector",
                             command=lambda: controller.show_frame(PageSeven))
        button7.pack(pady=10, padx=10)

        button8 = ttk.Button(MachineLFrame, text="Logistic Regression",
                             command=lambda: controller.show_frame(PageEight))
        button8.pack(pady=10, padx=10)

        button9 = ttk.Button(MachineLFrame, text="Random Forest",
                             command=lambda: controller.show_frame(PageNine))
        button9.pack(pady=10, padx=10)

        ComparisonFrame = tk.LabelFrame(self, text="Machine Learning Algorithms", font=LARGE_FONT, background="snow",
                                        fg="light sky blue")
        ComparisonFrame.pack(pady=10, padx=10, side=LEFT)

        button10 = ttk.Button(ComparisonFrame, text="Comparision With Pre-processing",
                              command=lambda: controller.show_frame(PageTen))
        button10.pack(pady=10, padx=10)

        button11 = ttk.Button(ComparisonFrame, text="Comparision Without Pre-processing",
                              command=lambda: controller.show_frame(PageEleven))
        button11.pack(pady=10, padx=10)

        AnalyseFrame = tk.LabelFrame(self, text="Analyse New Data", font=LARGE_FONT, background="snow",
                                     fg="light sky blue")
        AnalyseFrame.pack(pady=10, padx=10, side=LEFT)

        button12 = ttk.Button(AnalyseFrame, text="GUI",
                              command=lambda: controller.show_frame(PageTwelve))
        button12.pack(pady=10, padx=10)

        button13 = ttk.Button(aboutFrame, text="Analysis of Review Ratings",
                              command=lambda: controller.show_frame(PageThirteen))
        button13.pack(pady=10, padx=10)


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="snow")
        titleLabel = tk.Label(self, text="Review Spam Detection Using Machine Learning Algorithm", font=("Cambria", 20),
                              background="snow", fg="deep sky blue")
        titleLabel.pack(pady=10, padx=10)
        label = tk.Label(self, text="Know About Your Data: Word Cloud of Dataset", font=SMALL_FONT, fg="brown1",
                         bg="snow")
        label.pack(pady=5, padx=5)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(pady=5, padx=5)
        myImage = ImageTk.PhotoImage(pl.Image.open(Directory + "/WorldCloudBeforPre.png"))
        myLabel = tk.Label(self, image=myImage)
        myLabel.image = myImage
        myLabel.pack(pady=5, padx=5)


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="snow")
        titleLabel = tk.Label(self, text="Review Spam Detection Using Machine Learning Algorithm", font=("Cambria", 20),
                              background="snow", fg="deep sky blue")
        titleLabel.pack(pady=10, padx=10)

        label = tk.Label(self, text="Know About Your Data: Word Count of Most Frequent Words of Dataset",
                         font=SMALL_FONT, fg="brown1", bg="snow")
        label.pack(pady=5, padx=5)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(pady=5, padx=5)

        count1 = Counter(" ".join(data[data['Deceptive'] == 0]["Text"]).split()).most_common(35)
        df1 = pd.DataFrame.from_dict(count1)
        df1 = df1.rename(columns={0: "words in non-spam", 1: "count"})

        figure3 = Figure(figsize=(8, 8), dpi=100)
        ax3 = figure3.add_subplot(111)
        canvas3 = FigureCanvasTkAgg(figure3, self)
        canvas3.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT)
        df1.plot(kind='bar', legend=False, ax=ax3, color="green")
        ax3.set_xticklabels(df1["words in non-spam"])
        ax3.set_title('More frequent words in truthful Hotel Reviews')
        canvas3.draw()

        count2 = Counter(" ".join(data[data['Deceptive'] == 1]["Text"]).split()).most_common(35)
        df2 = pd.DataFrame.from_dict(count2)
        df2 = df2.rename(columns={0: "words in spam", 1: "count_"})

        figure4 = Figure(figsize=(8, 8), dpi=100)
        ax4 = figure4.add_subplot(111)
        canvas4 = FigureCanvasTkAgg(figure4, self)
        canvas4.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT)
        df2.plot(kind='bar', legend=False, ax=ax4, color="tomato")
        ax4.set_xticklabels(df2["words in spam"])
        ax4.set_title('More frequent words in Deceptive Hotel Reviews')
        canvas4.draw()


class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="snow")
        titleLabel = tk.Label(self, text="Review Spam Detection Using Machine Learning Algorithm", font=("Cambria", 20),
                              background="snow", fg="deep sky blue")
        titleLabel.pack(pady=10, padx=10)

        label = tk.Label(self, text="Know ABout Your Data: Data Distribution!", font=SMALL_FONT, fg="brown1", bg="snow")
        label.pack(pady=5, padx=5)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(pady=5, padx=5)

        figure1 = Figure(figsize=(8, 8), dpi=100)
        ax1 = figure1.add_subplot(111)
        canvas1 = FigureCanvasTkAgg(figure1, self)
        canvas1.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT)
        Data1 = pd.value_counts(data["Deceptive"], sort=True)
        Data1.plot(kind='bar', color=["tomato", "green"], ax=ax1)
        ax1.set_title('Distribution Of Data')
        canvas1.draw()

        figure2 = Figure(figsize=(8, 8), dpi=100)
        ax2 = figure2.add_subplot(111)
        canvas2 = FigureCanvasTkAgg(figure2, self)
        canvas2.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT)
        Data2 = pd.value_counts(data["Deceptive"], sort=True)
        Data2.plot(kind='pie', autopct='%1.0f%%', ax=ax2)
        ax2.set_title('Distribution Of Data')
        canvas2.draw()


class PageFour(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="snow")
        titleLabel = tk.Label(self, text="Review Spam Detection Using Machine Learning Algorithm", font=("Cambria", 20),
                              background="snow", fg="deep sky blue")
        titleLabel.pack(pady=10, padx=10)

        label = tk.Label(self, text="Pre-Processing Stage: Word Count", font=SMALL_FONT, fg="brown1", bg="snow")
        label.pack(pady=5, padx=5)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(pady=5, padx=5)

        data = pd.read_csv('ReviewsHotel.csv', encoding='latin-1')
        TextData = pd.read_csv('PreProcessedData.csv')
        TextData = pd.DataFrame(TextData)
        TextData["Deceptive"] = data["Deceptive"]
        data = TextData
        count1 = Counter(" ".join(data[data['Deceptive'] == 0]["Text"]).split()).most_common(35)
        df1 = pd.DataFrame.from_dict(count1)
        df1 = df1.rename(columns={0: "words in non-spam", 1: "count"})

        figure3 = Figure(figsize=(8, 8), dpi=100)
        ax3 = figure3.add_subplot(111)
        canvas3 = FigureCanvasTkAgg(figure3, self)
        canvas3.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT)
        df1.plot(kind='bar', legend=False, ax=ax3, color="green")
        ax3.set_xticklabels(df1["words in non-spam"])
        ax3.set_title('More frequent words in truthful Hotel Reviews')
        canvas3.draw()

        count2 = Counter(" ".join(data[data['Deceptive'] == 1]["Text"]).split()).most_common(35)
        df2 = pd.DataFrame.from_dict(count2)
        df2 = df2.rename(columns={0: "words in spam", 1: "count_"})

        figure4 = Figure(figsize=(8, 8), dpi=100)
        ax4 = figure4.add_subplot(111)
        canvas4 = FigureCanvasTkAgg(figure4, self)
        canvas4.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT)
        df2.plot(kind='bar', legend=False, ax=ax4, color="red")
        ax4.set_xticklabels(df2["words in spam"])
        ax4.set_title('More frequent words in Deceptive Hotel Reviews')
        canvas4.draw()


class PageFive(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="snow")
        titleLabel = tk.Label(self, text="Review Spam Detection Using Machine Learning Algorithm", font=("Cambria", 20),
                              background="snow", fg="deep sky blue")
        titleLabel.pack(pady=10, padx=10)

        label = tk.Label(self, text="Pre-Processing Stage: Word Cloud", font=SMALL_FONT, fg="brown1", bg="snow")
        label.pack(pady=5, padx=5)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(pady=5, padx=5)

        myImage = ImageTk.PhotoImage(pl.Image.open(Directory + "/WorldCloudAfterPre.png"))
        myLabel = tk.Label(self, image=myImage)
        myLabel.image = myImage
        myLabel.pack()


class PageSix(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="snow")
        titleLabel = tk.Label(self, text="Review Spam Detection Using Machine Learning Algorithm", font=("Cambria", 20),
                              background="snow", fg="deep sky blue")
        titleLabel.pack(pady=10, padx=10)

        label = tk.Label(self, text="Naive Bayes Machine Learning Algorithm", font=SMALL_FONT, fg="brown1", bg="snow")
        label.pack(pady=5, padx=5)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(pady=5, padx=5)

        # Naive Bayes With Preprocessed Data

        f = feature_extraction.text.CountVectorizer(stop_words='english')
        X = f.fit_transform(TextData["Text"])
        X = X.toarray()
        rating = np.array(data.Rating)
        X = np.column_stack((X, rating))

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['Deceptive'], test_size=0.33,
                                                                            random_state=42)
        bayes = naive_bayes.MultinomialNB()
        bayes.fit(X_train, y_train)

        global score_test_bayes_pre
        score_test_bayes_pre = bayes.score(X, data["Deceptive"])
        global recall_test_bayes_pre
        recall_test_bayes_pre = metrics.recall_score(data["Deceptive"], bayes.predict(X))
        global precision_test_bayes_pre
        precision_test_bayes_pre = metrics.precision_score(data["Deceptive"], bayes.predict(X))

        Data1 = {
            'measures': ['Accuracy', 'Recall', 'Precision'],
            'performance': [round(score_test_bayes_pre * 100, 2), round(recall_test_bayes_pre * 100, 2),
                            round(precision_test_bayes_pre * 100, 2)]
        }
        performanceTicks = [round(score_test_bayes_pre * 100, 2), round(recall_test_bayes_pre * 100, 2),
                            round(precision_test_bayes_pre * 100, 2)]

        NaiveBayesWithDF = DataFrame(Data1, columns=['measures', 'performance'])

        figure1 = Figure(figsize=(8, 8), dpi=100)
        ax1 = figure1.add_subplot(111)
        canvas1 = FigureCanvasTkAgg(figure1, self)
        canvas1.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT)
        NaiveBayesWithDF.plot(kind='bar', color="bisque", legend=False, ax=ax1)
        ax1.set_xticklabels(performanceTicks)
        ax1.set_title('With Pre Processed Data')
        ax1.set_xlabel('Accuracy     Recall      Precision')
        ax1.set_ylim(0, 100)
        canvas1.draw()

        # Naive Bayes Without Preprocessed Data

        f = feature_extraction.text.CountVectorizer()
        X = f.fit_transform(data["Text"])
        X = X.toarray()
        rating = np.array(data.Rating)
        X = np.column_stack((X, rating))

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['Deceptive'], test_size=0.33,
                                                                            random_state=42)

        bayes = naive_bayes.MultinomialNB()
        bayes.fit(X_train, y_train)

        global score_test_bayes
        score_test_bayes = bayes.score(X, data["Deceptive"])
        global recall_test_bayes
        recall_test_bayes = metrics.recall_score(data["Deceptive"], bayes.predict(X))
        global precision_test_bayes
        precision_test_bayes = metrics.precision_score(data["Deceptive"], bayes.predict(X))

        Data1 = {
            'measures': ['Accuracy', 'Recall', 'Precision'],
            'performance': [round(score_test_bayes * 100, 2), round(recall_test_bayes * 100, 2),
                            round(precision_test_bayes * 100, 2)]
        }

        performanceTicks = [round(score_test_bayes * 100, 2), round(recall_test_bayes * 100, 2),
                            round(precision_test_bayes * 100, 2)]

        NaiveBayesWithoutDF = DataFrame(Data1, columns=['measures', 'performance'])

        figure1 = Figure(figsize=(8, 8), dpi=100)
        ax1 = figure1.add_subplot(111)
        canvas1 = FigureCanvasTkAgg(figure1, self)
        canvas1.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT)
        NaiveBayesWithoutDF.plot(kind='bar', color="palegreen", legend=False, ax=ax1)
        ax1.set_xticklabels(performanceTicks)
        ax1.set_title('Without Pre Processed Data')
        ax1.set_xlabel('Accuracy     Recall      Precision')
        ax1.set_ylim(0, 100)
        canvas1.draw()


class PageSeven(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="snow")
        titleLabel = tk.Label(self, text="Review Spam Detection Using Machine Learning Algorithm", font=("Cambria", 20),
                              background="snow", fg="deep sky blue")
        titleLabel.pack(pady=10, padx=10)

        label = tk.Label(self, text="Support Vector Machine Learning Algorithm ", font=SMALL_FONT, fg="brown1",
                         bg="snow")
        label.pack(pady=5, padx=5)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(pady=5, padx=5)
        # Support Vector With Preprocessed Data

        f = feature_extraction.text.CountVectorizer(stop_words='english')
        X = f.fit_transform(TextData["Text"])
        X = X.toarray()
        rating = np.array(data.Rating)
        X = np.column_stack((X, rating))

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['Deceptive'], test_size=0.33,
                                                                            random_state=42)

        svc = svm.SVC()
        svc.fit(X_train, y_train)
        global score_test_svc_pre
        score_test_svc_pre = svc.score(X, data["Deceptive"])
        global recall_test_svc_pre
        recall_test_svc_pre = metrics.recall_score(data["Deceptive"], svc.predict(X))
        global precision_test_svc_pre
        precision_test_svc_pre = metrics.precision_score(data["Deceptive"], svc.predict(X))

        Data1 = {
            'measures': ['Accuracy', 'Recall', 'Precision'],
            'performance': [round(score_test_svc_pre * 100, 2), round(recall_test_svc_pre * 100, 2),
                            round(precision_test_svc_pre * 100, 2)]
        }
        performanceTicks = [round(score_test_svc_pre * 100, 2), round(recall_test_svc_pre * 100, 2),
                            round(precision_test_svc_pre * 100, 2)]

        SupportVectorWithDF = DataFrame(Data1, columns=['measures', 'performance'])

        figure1 = Figure(figsize=(8, 8), dpi=100)
        ax1 = figure1.add_subplot(111)
        canvas1 = FigureCanvasTkAgg(figure1, self)
        canvas1.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT)
        SupportVectorWithDF.plot(kind='bar', color="bisque", legend=False, ax=ax1)
        ax1.set_xticklabels(performanceTicks)
        ax1.set_title('With Preprocessed Data')
        ax1.set_xlabel('Accuracy     Recall      Precision')
        ax1.set_ylim(0, 100)
        canvas1.draw()

        # Support Vector Machine Without Preprocessed Data

        f = feature_extraction.text.CountVectorizer()
        X = f.fit_transform(data["Text"])
        X = X.toarray()
        rating = np.array(data.Rating)
        X = np.column_stack((X, rating))

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['Deceptive'], test_size=0.33,
                                                                            random_state=42)

        svc.fit(X_train, y_train)

        global score_test_svc
        score_test_svc = svc.score(X, data["Deceptive"])
        global recall_test_svc
        recall_test_svc = metrics.recall_score(data["Deceptive"], svc.predict(X))
        global precision_test_svc
        precision_test_svc = metrics.precision_score(data["Deceptive"], svc.predict(X))

        Data1 = {
            'measures': ['Accuracy', 'Recall', 'Precision'],
            'performance': [round(score_test_svc * 100, 2), round(recall_test_svc * 100, 2),
                            round(precision_test_svc * 100, 2)]
        }
        performanceTicks = [round(score_test_svc * 100, 2), round(recall_test_svc * 100, 2),
                            round(precision_test_svc * 100, 2)]

        SupportVectorWithoutDF = DataFrame(Data1, columns=['measures', 'performance'])

        figure1 = Figure(figsize=(8, 8), dpi=100)
        ax1 = figure1.add_subplot(111)
        canvas1 = FigureCanvasTkAgg(figure1, self)
        canvas1.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT)
        SupportVectorWithoutDF.plot(kind='bar', color="palegreen", legend=False, ax=ax1)
        ax1.set_xticklabels(performanceTicks)
        ax1.set_title('Without Pre Processed Data')
        ax1.set_xlabel('Accuracy     Recall      Precision')
        ax1.set_ylim(0, 100)
        canvas1.draw()


class PageEight(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="snow")
        titleLabel = tk.Label(self, text="Review Spam Detection Using Machine Learning Algorithm", font=("Cambria", 20),
                              background="snow", fg="deep sky blue")
        titleLabel.pack(pady=10, padx=10)

        label = tk.Label(self, text="Logistic Regression Machine Learning Algorithm ", font=SMALL_FONT, fg="brown1",
                         bg="snow")
        label.pack(pady=5, padx=5)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(pady=5, padx=5)
        # Logistic Regression With Preprocessed Data

        f = feature_extraction.text.CountVectorizer(stop_words='english')
        X = f.fit_transform(TextData["Text"])
        X = X.toarray()
        rating = np.array(data.Rating)
        X = np.column_stack((X, rating))

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['Deceptive'], test_size=0.33,
                                                                            random_state=42)

        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)

        global score_test_logreg_pre
        score_test_logreg_pre = logreg.score(X, data["Deceptive"])
        global recall_test_logreg_pre
        recall_test_logreg_pre = metrics.recall_score(data["Deceptive"], logreg.predict(X))
        global precision_test_logreg_pre
        precision_test_logreg_pre = metrics.precision_score(data["Deceptive"], logreg.predict(X))

        Data1 = {
            'measures': ['Accuracy', 'Recall', 'Precision'],
            'performance': [round(score_test_logreg_pre * 100, 2), round(recall_test_logreg_pre * 100, 2),
                            round(precision_test_svc_pre * 100, 2)]
        }
        performanceTicks = [round(score_test_logreg_pre * 100, 2), round(recall_test_logreg_pre * 100, 2),
                            round(precision_test_logreg_pre * 100, 2)]

        logregWithDF = DataFrame(Data1, columns=['measures', 'performance'])

        figure1 = Figure(figsize=(8, 8), dpi=100)
        ax1 = figure1.add_subplot(111)
        canvas1 = FigureCanvasTkAgg(figure1, self)
        canvas1.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT)
        logregWithDF.plot(kind='bar', color="bisque", legend=False, ax=ax1)
        ax1.set_xticklabels(performanceTicks)
        ax1.set_title('With PreProcessed Data')
        ax1.set_xlabel('Accuracy     Recall      Precision')
        ax1.set_ylim(0, 100)
        canvas1.draw()

        # Logistic Regression Without Preprocessed Data

        f = feature_extraction.text.CountVectorizer()
        X = f.fit_transform(data["Text"])
        X = X.toarray()
        rating = np.array(data.Rating)
        X = np.column_stack((X, rating))

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['Deceptive'], test_size=0.33,
                                                                            random_state=42)
        logreg.fit(X_train, y_train)

        global score_test_logreg
        score_test_logreg = logreg.score(X, data["Deceptive"])
        global recall_test_logreg
        recall_test_logreg = metrics.recall_score(data["Deceptive"], logreg.predict(X))
        global precision_test_logreg
        precision_test_logreg = metrics.precision_score(data["Deceptive"], logreg.predict(X))

        Data1 = {
            'measures': ['Accuracy', 'Recall', 'Precision'],
            'performance': [round(score_test_logreg * 100, 2), round(recall_test_logreg * 100, 2),
                            round(precision_test_svc * 100, 2)]
        }
        performanceTicks = [round(score_test_logreg * 100, 2), round(recall_test_logreg * 100, 2),
                            round(precision_test_logreg * 100, 2)]

        logregWithoutDF = DataFrame(Data1, columns=['measures', 'performance'])

        figure1 = Figure(figsize=(8, 8), dpi=100)
        ax1 = figure1.add_subplot(111)
        canvas1 = FigureCanvasTkAgg(figure1, self)
        canvas1.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT)
        logregWithoutDF.plot(kind='bar', color="palegreen", legend=False, ax=ax1)
        ax1.set_xticklabels(performanceTicks)
        ax1.set_title('Without Pre-Processing')
        ax1.set_xlabel('Accuracy     Recall      Precision')
        ax1.set_ylim(0, 100)
        canvas1.draw()


class PageNine(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="snow")
        titleLabel = tk.Label(self, text="Review Spam Detection Using Machine Learning Algorithm", font=("Cambria", 20),
                              background="snow", fg="deep sky blue")
        titleLabel.pack(pady=10, padx=10)

        label = tk.Label(self, text="Random Forest Machine Learning Algorithm ", font=SMALL_FONT, fg="brown1",
                         bg="snow")
        label.pack(pady=5, padx=5)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(pady=5, padx=5)
        # Random Forest With Preprocessed Data

        f = feature_extraction.text.CountVectorizer(stop_words='english')
        X = f.fit_transform(TextData["Text"])
        X = X.toarray()
        rating = np.array(data.Rating)
        X = np.column_stack((X, rating))

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['Deceptive'], test_size=0.33,
                                                                            random_state=42)

        randomF = RandomForestClassifier()
        randomF.fit(X_train, y_train)

        global score_test_randomF_pre
        score_test_randomF_pre = randomF.score(X, data["Deceptive"])
        global recall_test_randomF_pre
        recall_test_randomF_pre = metrics.recall_score(data["Deceptive"], randomF.predict(X))
        global precision_test_randomF_pre
        precision_test_randomF_pre = metrics.precision_score(data["Deceptive"], randomF.predict(X))

        Data1 = {
            'measures': ['Accuracy', 'Recall', 'Precision'],
            'performance': [round(score_test_randomF_pre * 100, 2), round(recall_test_randomF_pre * 100, 2),
                            round(precision_test_svc_pre * 100, 2)]
        }
        performanceTicks = [round(score_test_randomF_pre * 100, 2), round(recall_test_randomF_pre * 100, 2),
                            round(precision_test_randomF_pre * 100, 2)]

        randomFWithDF = DataFrame(Data1, columns=['measures', 'performance'])

        figure1 = Figure(figsize=(8, 8), dpi=100)
        ax1 = figure1.add_subplot(111)
        canvas1 = FigureCanvasTkAgg(figure1, self)
        canvas1.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT)
        randomFWithDF.plot(kind='bar', color="bisque", legend=False, ax=ax1)
        ax1.set_xticklabels(performanceTicks)
        ax1.set_title('With Preprocessing')
        ax1.set_xlabel('Accuracy     Recall      Precision')
        ax1.set_ylim(0, 100)
        canvas1.draw()

        # Random Forest Without Preprocessed Data

        f = feature_extraction.text.CountVectorizer()
        X = f.fit_transform(data["Text"])
        X = f.fit_transform(TextData["Text"])
        X = X.toarray()
        rating = np.array(data.Rating)
        X = np.column_stack((X, rating))

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['Deceptive'], test_size=0.33,
                                                                            random_state=42)

        randomF = RandomForestClassifier()
        randomF.fit(X_train, y_train)
        global score_test_randomF
        score_test_randomF = randomF.score(X, data["Deceptive"])
        global recall_test_randomF
        recall_test_randomF = metrics.recall_score(data["Deceptive"], randomF.predict(X))
        global precision_test_randomF
        precision_test_randomF = metrics.precision_score(data["Deceptive"], randomF.predict(X))

        Data1 = {
            'measures': ['Accuracy', 'Recall', 'Precision'],
            'performance': [round(score_test_randomF * 100, 2), round(recall_test_randomF * 100, 2),
                            round(precision_test_randomF * 100, 2)]
        }
        performanceTicks = [round(score_test_randomF * 100, 2), round(recall_test_randomF * 100, 2),
                            round(precision_test_randomF * 100, 2)]

        randomFWithoutDF = DataFrame(Data1, columns=['measures', 'performance'])

        figure1 = Figure(figsize=(8, 8), dpi=100)
        ax1 = figure1.add_subplot(111)
        canvas1 = FigureCanvasTkAgg(figure1, self)
        canvas1.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT)
        randomFWithoutDF.plot(kind='bar', color="palegreen", legend=False, ax=ax1)
        ax1.set_xticklabels(performanceTicks)
        ax1.set_title('Without Pre Processing')
        ax1.set_xlabel('Accuracy     Recall      Precision')
        ax1.set_ylim(0, 100)
        canvas1.draw()


class PageTen(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="snow")
        titleLabel = tk.Label(self, text="Review Spam Detection Using Machine Learning Algorithm", font=("Cambria", 20),
                              background="snow", fg="deep sky blue")
        titleLabel.pack(pady=10, padx=10)

        label = tk.Label(self, text="Comparision Of Algorithm With Pre Processed Data", font=SMALL_FONT, fg="brown1",
                         bg="snow")
        label.pack(pady=5, padx=5)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(pady=5, padx=5)

        bayes_accuracy = [score_test_bayes_pre, recall_test_bayes_pre, precision_test_bayes_pre]

        svc_accuracy = [score_test_svc_pre, recall_test_svc_pre, precision_test_svc_pre]

        logreg_accuracy = [score_test_logreg_pre, recall_test_logreg_pre, precision_test_logreg_pre]

        randomF_accuracy = [score_test_randomF_pre, recall_test_randomF_pre, precision_test_randomF_pre]

        print(bayes_accuracy)
        print(svc_accuracy)
        print(logreg_accuracy)
        print(randomF_accuracy)

        # Create bars
        barWidth = 1

        # The X position of bars
        r1 = [1, 6, 11]
        r2 = [2, 7, 12]
        r3 = [3, 8, 13]
        r4 = [4, 9, 14]

        performanceTicks = (round(score_test_bayes_pre * 100, 2), round(score_test_svc_pre * 100, 2),
                            round(score_test_logreg_pre * 100, 2), round(score_test_randomF_pre * 100, 2), " ",
                            round(recall_test_bayes_pre * 100, 2), round(recall_test_svc_pre * 100, 2),
                            round(recall_test_logreg_pre * 100, 2), round(recall_test_randomF_pre * 100, 2), " ",
                            round(precision_test_bayes_pre * 100, 2), round(precision_test_svc_pre * 100, 2),
                            round(precision_test_logreg_pre * 100, 2), round(precision_test_randomF_pre * 100, 2))

        # Create barplot
        figure1 = Figure(figsize=(10, 15), dpi=100)
        ax1 = figure1.add_subplot(111)
        canvas1 = FigureCanvasTkAgg(figure1, self)
        canvas1.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT)

        ax1.bar(r1, bayes_accuracy, width=barWidth, color=(0.3, 0.1, 0.4, 0.6), label='Bayes')
        ax1.bar(r2, svc_accuracy, width=barWidth, color=(0.3, 0.5, 0.4, 0.6), label='Support Vector Machine')
        ax1.bar(r3, logreg_accuracy, width=barWidth, color=(0.3, 0.9, 0.4, 0.6), label='Logistic Regression')
        ax1.bar(r4, randomF_accuracy, width=barWidth, color=(0.3, 0.7, 0.4, 0.6), label='Random Forest')

        ax1.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        ax1.set_xticklabels(performanceTicks, rotation='vertical')  # position=[r1,r2,r3,r4])
        ax1.set_ylabel('Accuracy')

        ax1.legend()
        ax1.set_ylim(0, 1.2)

        canvas1.draw()


class PageEleven(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="snow")
        titleLabel = tk.Label(self, text="Review Spam Detection Using Machine Learning Algorithm", font=("Cambria", 20),
                              background="snow", fg="deep sky blue")
        titleLabel.pack(pady=10, padx=10)

        label = tk.Label(self, text="Comparision Of Algorithm Without Pre Processed Data", font=SMALL_FONT, fg="brown1",
                         bg="snow")
        label.pack(pady=5, padx=5)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(pady=5, padx=5)

        bayes_accuracy = [score_test_bayes, recall_test_bayes, precision_test_bayes]
        svc_accuracy = [score_test_svc, recall_test_svc, precision_test_svc]
        logreg_accuracy = [score_test_logreg, recall_test_logreg, precision_test_logreg]
        randomF_accuracy = [score_test_randomF, recall_test_randomF, precision_test_randomF]

        print(bayes_accuracy)
        print(svc_accuracy)
        print(logreg_accuracy)
        print(randomF_accuracy)

        # Create bars
        barWidth = 1

        # The X position of bars
        r1 = [1, 6, 11]
        r2 = [2, 7, 12]
        r3 = [3, 8, 13]
        r4 = [4, 9, 14]

        performanceTicks = (round(score_test_bayes * 100, 2), round(score_test_svc * 100, 2),
                            round(score_test_logreg * 100, 2), round(score_test_randomF * 100, 2), " ",
                            round(recall_test_bayes * 100, 2), round(recall_test_svc * 100, 2),
                            round(recall_test_logreg * 100, 2), round(recall_test_randomF * 100, 2), " ",
                            round(precision_test_bayes * 100, 2), round(precision_test_svc * 100, 2),
                            round(precision_test_logreg * 100, 2), round(precision_test_randomF * 100, 2))

        # Create barplot
        figure1 = Figure(figsize=(10, 15), dpi=100)
        ax1 = figure1.add_subplot(111)
        canvas1 = FigureCanvasTkAgg(figure1, self)
        canvas1.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT)

        ax1.bar(r1, bayes_accuracy, width=barWidth, color=(0.3, 0.1, 0.4, 0.6), label='Bayes')
        ax1.bar(r2, svc_accuracy, width=barWidth, color=(0.3, 0.5, 0.4, 0.6), label='Support Vector Machine')
        ax1.bar(r3, logreg_accuracy, width=barWidth, color=(0.3, 0.9, 0.4, 0.6), label='Logistic Regression')
        ax1.bar(r4, randomF_accuracy, width=barWidth, color=(0.3, 0.7, 0.4, 0.6), label='Random Forest')

        ax1.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        ax1.set_xticklabels(performanceTicks, rotation='vertical')  # position=[r1,r2,r3,r4])
        ax1.set_ylabel('Accuracy')

        ax1.legend()
        ax1.set_ylim(0, 1.2)

        canvas1.draw()


class PageTwelve(tk.Frame):
    def __init__(self, parent, controller):

        tk.Frame.__init__(self, parent, bg="snow")
        titleLabel = tk.Label(self, text="Review Spam Detection Using Machine Learning Algorithm", font=("Cambria", 20),
                              background="snow", fg="deep sky blue")
        titleLabel.pack(pady=10, padx=10)

        label = tk.Label(self, text="Predict New Review ", font=SMALL_FONT, fg="brown1")
        label.pack(pady=5, padx=5)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(pady=5, padx=5)

        def append_list_as_row():
            review = str(reviewEntry.get(1.0, END))

            with open('ReviewsHotel.csv', 'a', newline='') as write_obj:
                # Create a writer object from csv module
                csv_writer = writer(write_obj, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
                # Add contents of list as last row in the csv file
                csv_writer.writerow([1, HotelNameClicked.get(), HotelRatingClicked.get(), 'MTurk', review])

        def ShowResult():
            data1 = pd.read_csv('ReviewsHotel.csv', encoding='latin-1')
            f = feature_extraction.text.CountVectorizer()
            X = f.fit_transform(data1["Text"])
            X = X.toarray()
            rating = np.array(data1.Rating)
            print(len(rating))
            X = np.column_stack((X, rating))

            print(np.shape(X))
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X[0:1600, ], data1.Deceptive[0:1600],
                                                                                test_size=0.33,
                                                                                random_state=42)
            randomF = RandomForestClassifier()
            randomF.fit(X_train, y_train)

            print(randomF.predict(X[[-1], :]))

            result = randomF.predict(X[[-1], :])

            global myLabel
            if result == 1:
                myLabel = tk.Label(self, text="Deceptive Review", bg="snow", font=LARGE_FONT, fg="red2")
                myLabel.pack(pady=10, padx=10)
            else:
                myLabel = tk.Label(self, text="Truthful Review", bg="snow", font=LARGE_FONT, fg="green2")
                myLabel.pack(pady=10, padx=10)

        def DeleteAll():
            HotelRatingClicked.set("3")
            HotelNameClicked.set("affinia")
            reviewEntry.delete('1.0', END)
            myLabel.pack_forget()

        Label1 = tk.Label(self, text="Write a Review", bg="snow", font=LARGE_FONT)
        Label1.pack(pady=5, padx=5)

        HotelNameClicked = StringVar()
        HotelNameClicked.set("affinia")
        hotelNamesDropDown = OptionMenu(self, HotelNameClicked, "affinia", "allegro", "amalfi", "ambassador", "conrad",
                                        "fairmont", "knickerbocker", "omni", "hardrock", "hilton", "homewood", "hyatt",
                                        "intercontinental", "james", "monaco", "palmer", "sheraton", "sofitel",
                                        "swissotel",
                                        "talbott")

        hotelNamesDropDown.pack()

        reviewEntry = tk.Text(self, font=SMALL_FONT, height=12, width=35)
        reviewEntry.pack(pady=5, padx=5)

        HotelRatingClicked = IntVar()
        HotelRatingClicked.set("3")
        HotelRatingDropDown = OptionMenu(self, HotelRatingClicked, "1", "2", "3", "4", "5")
        HotelRatingDropDown.pack()

        myButton = tk.Button(self, text="Submit", command=append_list_as_row, font=SMALL_FONT)
        myButton.pack(pady=5, padx=5)

        myshowButton = tk.Button(self, text="Show", command=ShowResult, font=SMALL_FONT)
        myshowButton.pack(pady=5, padx=5)

        reviewOtherHotelButton = tk.Button(self, text="Review Another Hotel", command=DeleteAll, font=SMALL_FONT)
        reviewOtherHotelButton.pack(pady=5, padx=5)

        # myLabel = tk.Label(self, text=" ", bg="snow", font=SMALL_FONT)
        # myLabel.pack(pady=5, padx=5)


class PageThirteen(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="snow")
        titleLabel = tk.Label(self, text="Review Spam Detection Using Machine Learning Algorithm", font=("Cambria", 20),
                              background="snow", fg="deep sky blue")
        titleLabel.pack(pady=10, padx=10)
        label = tk.Label(self, text="Know About Your Data: Analysis of Review Ratings", font=SMALL_FONT, fg="brown1",
                         bg="snow")
        label.pack(pady=5, padx=5)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack(pady=5, padx=5)

        DeceptiveDF = pd.DataFrame(data[data.Deceptive == 1].groupby(['Rating']).count())

        TruthfulDF = pd.DataFrame(data[data.Deceptive == 0].groupby(['Rating']).count())

        figure4 = Figure(figsize=(8, 8), dpi=100)
        ax = figure4.add_subplot(111)
        canvas = FigureCanvasTkAgg(figure4, self)
        canvas.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT)
        TruthfulDF.plot(kind='bar', legend=False, ax=ax, color="palegreen")
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        ax.set_title('Truthful Hotel Review Ratings Count')
        canvas.draw()

        figure3 = Figure(figsize=(8, 8), dpi=100)
        ax3 = figure3.add_subplot(111)
        canvas3 = FigureCanvasTkAgg(figure3, self)
        canvas3.get_tk_widget().pack(fill=tk.BOTH, side=tk.LEFT)
        DeceptiveDF.plot(kind='bar', legend=False, ax=ax3, color="bisque")
        ax3.set_xlabel('Rating')
        ax3.set_ylabel('Count')
        ax3.set_title('Deceptive Hotel Review Ratings Count')
        canvas3.draw()


app = SpamDetectionApp()
app.mainloop()
