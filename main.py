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
ALLOWED_EXTENSIONS = {'txt', 'csv','tsv','wav','mp3'}

app = Flask(__name__,static_url_path='/images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

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
        plot_t = request.form['plot_t']
        # check if the post request has the file part
        if 'file1' not in request.files or 'file2' not in request.files:
            return "No files Selected"
        file1 = request.files['file1']
        file2 = request.files['file2']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file1.filename == '' or file2.filename == '':
            return "No files Selected"
        if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
            #filename = secure_filename(file.filename)
            filename1='file1'
            filename2='file2'
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
            file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
            if plot_t=='interactive':
                if input_type=='series':
                    return redirect('/plot_series')
                if input_type=='audio':
                    return redirect('/plot_audio')
            if plot_t=='static':
                if input_type=='series':
                    return redirect('/plot_series_static')
                if input_type=='audio':
                    return redirect('/plot_audio_static')




@app.route('/plot_audio')
def plot_audio():
    #data = pd.read_csv('uploads/tst.tsv', 
    #               header=None,
    #               sep=',')

    #X = data.iloc[:, 1:]
    #y = data.iloc[:, 0]
    #fg=DTW(X.iloc[1, :], X.iloc[2, :]).plot(standard_graph=False,y_shift=6.5)
    fg=DTW('uploads/file1','uploads/file2',audio_files=True).plot(standard_graph=False)
    return render_template('plot.html',plot=fg)

@app.route('/plot_series')
def plot_series():
    file1 = pd.read_csv('uploads/file1', 
                   header=None,
                   sep=',')

    X1 = file1.iloc[:, 1:]
    y1 = file1.iloc[:, 0]

    file2 = pd.read_csv('uploads/file2', 
                   header=None,
                   sep=',')

    X2 = file2.iloc[:, 1:]
    y2 = file2.iloc[:, 0]


    fg=DTW(X1.iloc[1, :], X2.iloc[1, :]).plot(standard_graph=False)
    return render_template('plot.html',plot=fg)


@app.route('/plot_audio_static')
def plot_audio_static():
    fg=DTW('uploads/file1','uploads/file2',audio_files=True).plot(standard_graph=True)
    return render_template('plot.html',plot=fg)
    return render_template('static.html')



@app.route('/plot_series_static')
def plot_series_static():
    file1 = pd.read_csv('uploads/file1', 
                   header=None,
                   sep=',')

    X1 = file1.iloc[:, 1:]
    y1 = file1.iloc[:, 0]

    file2 = pd.read_csv('uploads/file2', 
                   header=None,
                   sep=',')

    X2 = file2.iloc[:, 1:]
    y2 = file2.iloc[:, 0]


    fg=DTW(X1.iloc[1, :], X2.iloc[1, :]).plot(standard_graph=True)
    return render_template('plot.html',plot=fg)




@app.route('/plot_test')
def plot_test():
    audio1 = 'audio_data/182503__swiftoid__birds-chirping-01-down-small-park-lane.wav'
    audio2 = 'audio_data/501589__tim-kahn__yellow-headed-blackbird.wav'
    fg=DTW(audio1, audio2, audio_files=True).plot(standard_graph=False) 
    return render_template('plot.html',plot=fg)




if __name__ == '__main__':
    app.run(host='localhost', port=0000)
