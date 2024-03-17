import pickle
import pandas as pd
a = [[0.000000 ,0.011111 ,0.000000 ,0.0 ,0 ,0.068965 ,0.017048,	0.023832,"May" ,1,0,"New_Visitor"]]
cols = ['BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Weekend',
       'Administrative_per_duration', 'Informational_per_duration',
       'ProductRel_per_dur', 'Month', 'Region', 'TrafficType', 'VisitorType']
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

model1 = pickle.load(open("online_shoppers_LR.pkl","rb"))
y_pred = model1.predict(X_test_r)
print("The customer will generate revenue: ", y_pred)



#model1.predict([[-0.219842, -0.846878,0.556120,-0.308907,1.808134,0.136177,-0.495174,-0.997299, 1.598950,0.0,0.0 , 0.0,0.0 ,0.0 , 0.0, 1.0 , 0.0 , 0.0,0.0 , 0.0 ,1.0]])
