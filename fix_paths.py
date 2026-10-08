import json
def fix_path(file, old, new):
    with open(file, 'r', encoding='utf-8') as f:
        data = f.read()
    data = data.replace(old, new)
    with open(file, 'w', encoding='utf-8') as f:
        f.write(data)

fix_path('notebooks/Chapter3_CNN_Diabetes.ipynb', '"diabetes_prediction_dataset.csv"', '"../data/diabetes_prediction_dataset.csv"')
fix_path('notebooks/Chapter3_CNN_VietnamHousing.ipynb', '"vietnam_housing_dataset.csv"', '"../data/vietnam_housing_dataset.csv"')
print("Done")
