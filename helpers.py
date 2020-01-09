import plotly
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import json

def create_plot():


    N = 40
    x = np.linspace(0, 1, N)
    y = np.random.randn(N)
    df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe

    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")

    data=[
        go.Scatter(x=df.Date,y=df['AAPL.High'],name="AAPL High",line_color='deepskyblue',opacity=0.8),
       go.Scatter( x=df.Date,y=df['AAPL.Low'],name="AAPL Low",line_color='dimgray',opacity=0.8)

   ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON
