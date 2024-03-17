from flask import Flask,request,render_template
import pickle
import pandas as pd

#intialize app
app = Flask(__name__)

# load the model
model1 = pickle.load(open("online_shoppers_LR.pkl","rb"))
      

transformer = pickle.load(open("transformer.pkl","rb"))
encoder = pickle.load(open("encoder.pkl","rb"))        


@app.route("/")
def hello():
    return render_template("form.html")
#@app.route("/", methods = ["POST"])
@app.route("/submit", methods = ["POST"])
# def predict():
def form_data():
    BounceRates = float(request.form['BounceRates'])
    ExitRates = float(request.form['ExitRates'])
    PageValues = float(request.form['PageValues'])
    SpecialDay = float(request.form['SpecialDay'])
    Weekend = float(request.form['Weekend'])
    Administrative_per_duration = float(request.form['Administrative_per_duration'])
    Informational_per_duration = float(request.form['Informational_per_duration'])
    ProductRel_per_dur = float(request.form['ProductRel_per_dur'])
    Month = request.form['Month']
    Region = float(request.form['Region'])  # Assuming Region is numeric
    TrafficType = float(request.form['TrafficType'])
    VisitorType = request.form['VisitorType']

    cols = ['BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Weekend', 'Administrative_per_duration', 'Informational_per_duration',
       'ProductRel_per_dur', 'Month', 'Region', 'TrafficType', 'VisitorType']
    a = [[BounceRates, ExitRates, PageValues, SpecialDay, Weekend,
       Administrative_per_duration, Informational_per_duration,
       ProductRel_per_dur, Month, Region, TrafficType, VisitorType]]
    X_test = pd.DataFrame(a,columns =cols )
    print(X_test)
    test_cat = X_test.select_dtypes(include = "object")
    test_num = X_test.select_dtypes(exclude = "object")
    transformer = pickle.load(open("transformer.pkl","rb"))
    encoder = pickle.load(open("encoder.pkl","rb"))
    X_test_scale = pd.DataFrame(transformer.transform(test_num),columns = test_num.columns)
    cols = encoder.get_feature_names_out(input_features=test_cat.columns)
    X_test_one_hot= pd.DataFrame(encoder.transform(test_cat).toarray(),columns=cols)
    X_test_r = pd.concat((X_test_scale,X_test_one_hot),axis=1)
        
    print("Input data:")
    print(X_test_r)
    y_pred = model1.predict(X_test_r)
    print("Predictions:")
    print(y_pred)
    return render_template("index.html", y_pred = f'customer will generate Revenue: {y_pred}')

        

if __name__ == "__main__":
    app.run(debug=True) 