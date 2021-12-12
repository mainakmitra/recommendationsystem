from flask import Flask, jsonify, render_template, request, redirect, url_for
import model
import pandas as pd
import numpy as np
import json
    
# Create the application.
app = Flask(__name__) 

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def loadResult():
    userid = request.form['username']
    df = model.getModeloutput(userid)
    pretty_json = json.dumps(df['name'].values.tolist(), indent=1)
    return pretty_json


if __name__ == '__main__':
    app.debug=True
    app.run()