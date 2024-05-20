import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv("NF50.csv")

# Splitting data in target and dependent feature
X = df[['high', 'low', 'close', 'shares_traded', 'turnover', 'day', 'month', 'year']]
Y = df['open_price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lr = LinearRegression()
lr.fit(X_train, Y_train)

pickle.dump(lr, open("model.pkl", "wb"))
