from flask import Flask,jsonify,request,Response
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import json
import os

app = Flask(__name__)
port = int(os.getenv("PORT", 3000))

iris = load_iris()
x = iris.data
y = iris.target
x_train,x_test,y_train,y_test= train_test_split(x,y)
rfc = RandomForestClassifier(n_estimators=100,n_jobs=2)
rfc.fit(x_train,y_train)

@app.route('/predict_api', methods=['POST'])
def predict():
    # Error checking
    req_body = request.get_json(force=True)
    # Convert JSON to numpy array
    sepal_length = req_body['sl']
    sepal_width = req_body['sw']
    petal_length = req_body['pl']
    petal_width = req_body['pw']

    iris_class = rfc.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    print(iris_class)
    if (iris_class[0] == 0):
        result = "Iris Setosa"
    elif (iris_class[0] == 1):
        result = "Iris Versicolor"
    elif (iris_class[0] == 2):
        result = "Iris Virginica"

    msg = {
        "message": "Your flower is %s" % (result)
    }
    resp = Response(response=json.dumps(msg),
                    status=200, \
                    mimetype="application/json")
    return resp

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=port)