from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)


#ruten for hjemmesiden
@app.route('/')
def hjemmeside():
    return render_template('hjemmeside.html')

#definerer ruten for prediksjoner
@app.route('/predict', methods=['POST'])
def home():
    #henter data
    data1 = float(request.form['a'])
    data2 = float(request.form['b'])
    data3 = float(request.form['c'])
    data4 = float(request.form['d'])
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    return render_template('etterside.html', data=pred)

#kjører programmet
if __name__ == "__main__":
    app.run(debug=True)
