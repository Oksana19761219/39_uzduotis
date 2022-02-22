from flask import Flask, request, render_template
import pickle

app = Flask(__name__)


with open('iris_predictor.pickle', 'rb') as f:
    model = pickle.load(f)

@app.route("/", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        sepal_length = int(request.form['sepal_length'])/10
        sepal_width = int(request.form['sepal_width'])/10
        petal_length = int(request.form['petal_length'])/10
        petal_width = int(request.form['petal_width'])/10

        predicted = model.predict([[sepal_length,
                                    sepal_width,
                                    petal_length,
                                    petal_width]])

        return render_template("index.html", predicted=predicted[0])
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(port=8000)

