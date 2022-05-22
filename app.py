import pandas as pd
import numpy as np
from gradient_descent import BatchGradientDescentRegressor
from flask import Flask, request, render_template
from flask_cors import cross_origin

app = Flask(__name__)


def preprocess(input_str):
    input_list = input_str.strip().split()
    return input_list


def input_check(input_list):
    try:
        input_list = [int(i) for i in input_list]
    except Exception as e:
        return e
    if len(input_list) < 5:
        return 'Enter at least 5 numbers'
    return "ok"


@app.route('/')
@cross_origin()
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
@cross_origin()
def predictor():
    input_str = request.form.get('seq')
    input_list = preprocess(input_str=input_str)
    is_error = input_check(input_list)
    if is_error == 'ok':
        input_list = [int(i) for i in input_list]
        y = pd.DataFrame({'x': input_list[:-1]})
        x_test = np.array(input_list[-1]).reshape(-1, 1)
        x = y.shift(1)
        data = pd.concat([x, y], axis=1)
        data.columns = ['x', 'y']
        data.dropna(inplace=True)

        x_train = data['x'].values.reshape(-1, 1)
        y_train = data['y'].values

        x_train_max = x_train.max()
        x_train_min = x_train.min()
        x_train = (x_train - x_train_min) / (x_train_max - x_train_min)  # min-max normalization

        gd = BatchGradientDescentRegressor()
        gd.fit(x_train, y_train)

        x_test = (x_test - x_train_min) / (x_train_max - x_train_min)

        if gd.abort == 0:
            result = f'{input_str.strip()} {str(int(np.round(gd.predict(x_test))))}'
        else:
            result = f'Too complex. Unable to predict.'
    else:
        result = is_error

    return render_template('index.html', log=result)


if __name__ == '__main__':
    app.run()
