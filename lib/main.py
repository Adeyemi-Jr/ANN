import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(tf.__version__)



#####################################
#               PART 1
#####################################

#Data preprocessing

df = pd.read_csv('../Churn_Modelling.csv')
X = df

y = df['Exited']
X.drop(['RowNumber','CustomerId','Surname','Exited'], inplace = True, axis = 1)


# Label encoding of gender
le = LabelEncoder()
X['Gender'] = le.fit_transform(X['Gender'])

# One hot encoding
X_dummy_variable = pd.get_dummies(df['Geography'],prefix= 'country')
X.drop('Geography', inplace = True, axis = 1)
X=X.join(X_dummy_variable)


#Splitting into Training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)




########################################################
#               PART 2 Building ANN
########################################################


#Initialisation the ANN
ann = tf.keras.models.Sequential()

#first layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

#second layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

#output layer
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


########################################################
#               PART 3 Training the ANN
########################################################

#compiling ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )


#Training the ANN on the Traninig set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)


########################################################
#               PART 4 Prediction
########################################################

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))> 0.5)

y_pred = ann.predict(X_test)

y_pred = y_pred > 0.5

#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))






