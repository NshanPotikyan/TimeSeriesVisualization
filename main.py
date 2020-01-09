from flask import Flask
from flask import render_template
from helpers import create_plot
app = Flask(__name__,static_url_path='/images')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot')
def plot():
    bar=create_plot()
    return render_template('plot.html',plot=bar)

if __name__ == '__main__':
    app.run(port=8080)
