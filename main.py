from flask import Flask
from flask import render_template, request,redirect,flash
#from helpers import create_plot
from dtw import DTW
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'csv','tsv'}

app = Flask(__name__,static_url_path='/images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        input_type = request.form['type']
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            #filename = secure_filename(file.filename)
            filename='tst.tsv'
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect('/plot_audio')


@app.route('/plot_series')
def plot_series():

    data = pd.read_csv('uploads/tst.tsv', 
                   header=None,
                   sep=',')

    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    fg=DTW(X.iloc[1, :], X.iloc[2, :]).plot(standard_graph=False,y_shift=6.5)
    return render_template('plot.html',plot=fg)

@app.route('/plot_audio')
def plot_audio():

    data = pd.read_csv('uploads/tst.tsv', 
                   header=None,
                   sep=',')

    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    fg=DTW(X.iloc[1, :], X.iloc[2, :]).plot(standard_graph=False,y_shift=6.5)
    return render_template('plot.html',plot=fg)



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80)
