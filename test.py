import pandas as pd
data = pd.read_csv('creditcard.csv')
print(data.columns)
print(data.tail(5))