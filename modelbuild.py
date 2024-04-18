# %%
import pandas as pd
df = pd.read_csv('advertising.csv')
df.head()

# %%
df.columns
# %%
df.info()
# %%
df = df.drop(['Unnamed: 0'], axis=1)
df.head()
# %%
# split data into data and target
X = df.loc[:, df.columns != 'Sales']
X.head()
# %%
y = df['Sales']
y.head()
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25)
print(X_train.shape)
print(X_train.head())
# %%
print(X_test.shape)
print(X_test.head())
# %%
print(y_train.shape)
print(y_train.head())
# %%
print(y_test.shape)
print(y_test.head())
# %%
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# %%
model.fit(X_train, y_train)
# %%
predictions = model.predict(X_test)
predictions[:10]
# %%
import joblib
joblib.dump(model,'lr_model.pkl')
# %%
import numpy as np
test_data = [159.1, 60.2, 90]
#Convert into numpy array and reshape
test_data = np.array(test_data).astype(np.float64)
test_data = test_data.reshape(1,-1)
test_data
# %%
filePath = 'lr_model.pkl'
file = open(filePath, "rb")
trained_model = joblib.load(file)
# %%
prediction = trained_model.predict(test_data)
print(prediction)
# %%
