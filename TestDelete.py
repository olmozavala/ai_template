import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing

mydata = {'x':[1,2,4],
          'y':[1,2,3]}

df = DataFrame(mydata)

print(df)
scaler = preprocessing.MinMaxScaler()
scaler.fit(df)
print(scaler.transform(df))
