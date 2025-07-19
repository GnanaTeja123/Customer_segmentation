import pandas as pd

def load_online_retail_data(filepath):
    data = pd.read_excel(filepath)
    data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]
    data.dropna(subset=['CustomerID'], inplace=True)
    data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
    return data