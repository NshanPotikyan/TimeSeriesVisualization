import plotly
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import json

def create_plot():



#    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")

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


    data=[
        go.Scatter(x=df.Date,y=df['AAPL.High'],name="AAPL High",line_color='deepskyblue',opacity=0.8),
       go.Scatter( x=df.Date,y=df['AAPL.Low'],name="AAPL Low",line_color='dimgray',opacity=0.8)

   ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON
