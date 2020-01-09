from flask import Flask
from flask import render_template
#from helpers import create_plot
from dtw import DTW
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__,static_url_path='/images')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot')
def plot():
#bar=create_plot()
    train = pd.read_csv('datasets/ECG200/ECG200_TRAIN.tsv', 
                   header=None,
                   sep='\t')

    test = pd.read_csv('datasets/ECG200/ECG200_TEST.tsv', 
                   header=None,
                   sep='\t')
    X_train = train.iloc[:, 1:]
    y_train = train.iloc[:, 0]
    X_test = test.iloc[:, 1:]
    y_test = test.iloc[:, 0]
    #return render_template('plot.html',plot=bar)
    #DTW(X_train.iloc[0, :], X_train.iloc[1, :]).plot(y_shift=6.5)
    fg=DTW(X_train.iloc[0, :], X_train.iloc[2, :]).plot(standard_graph=False)
    print(fg)
    return render_template('plot.html',plot=fg)

if __name__ == '__main__':
    app.run(port=8080)
