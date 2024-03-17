import pandas as pd
from sklearn import model_selection
import pickle

data = pd.read_csv("online_shoppers.csv")
data.drop(columns = ["Unnamed: 0"],axis=1,inplace=True)
y = data["Revenue"]
X = data.drop("Revenue",axis=1)
print(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 1)

#Split further into train_num and train_cat. Also test_num and test_cat
train_cat = X_train.select_dtypes(include = "object")
train_num = X_train.select_dtypes(exclude = "object")
test_cat = X_test.select_dtypes(include = "object")
test_num = X_test.select_dtypes(exclude = "object")

from sklearn.preprocessing import StandardScaler

transformer = StandardScaler().fit(train_num)
X_train_scale = pd.DataFrame(transformer.transform(train_num),columns = train_num.columns)
print(X_train_scale)
pickle.dump(transformer,open("transformer.pkl","wb"))
#X_test_scale = pd.DataFrame(transformer.transform(test_num),columns = test_num.columns)



# Encode the categorical features using One-Hot Encoding
# One hot encoding
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='error', drop='first')
encoder.fit(train_cat)
cols = encoder.get_feature_names_out(input_features=train_cat.columns)
#X_train_one_hot = encoder.transform(train_cat).toarray()
X_train_one_hot = pd.DataFrame(encoder.transform(train_cat).toarray(),columns=cols)

pickle.dump(encoder,open("encoder.pkl","wb"))
#X_test_one_hot= pd.DataFrame(encoder.transform(test_cat).toarray(),columns=cols)

#re-concatenate train_num and train_cat as X_train as well as test_num and test_cat as X_test
X_train_r = pd.concat((X_train_scale,X_train_one_hot),axis=1)
#X_test_r = pd.concat((X_test_scale,X_test_one_hot),axis=1)

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_r, y_train)


# from sklearn.ensemble import RandomForestClassifier
# RF = RandomForestClassifier(max_depth=2, random_state=0)
# RF.fit(X_train_resampled, y_train_resampled)
# y_pred_RF = RF.predict(X_test_r)

from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(X_train_resampled, y_train_resampled)

pd.set_option('display.max_columns', None)
 
#y_pred = model1.predict(X_test_r)

# from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy: ",accuracy)


# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# print("Precision:", precision)
# print("Recall:", recall)


# cm = confusion_matrix(y_test, y_pred)
# # Define labels for rows and columns
# labels = ['Actual 0', 'Actual 1']
# columns = ['Predicted 0', 'Predicted 1']

# # Create a DataFrame for the confusion matrix
# confusion_df = pd.DataFrame(cm, index=labels, columns=columns)

# # Print the confusion matrix
# print(confusion_df)

# #Get feature importances
# feature_importances = RF.feature_importances_
# feature_names = X_train_resampled.columns
# feature_imp_list = list(zip(feature_names, feature_importances))

# feature_imp_list.sort(key = lambda x: x[1],reverse = True) 
# print(feature_imp_list)

#save the model
pickle.dump(model1,open("online_shoppers_LR.pkl" , "wb"))