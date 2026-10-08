import nbformat

def fix_nb(file_path, old_path, new_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if old_path in cell.source:
                cell.source = cell.source.replace(old_path, new_path)
                print(f'Fixed {file_path}')
                
    with open(file_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

fix_nb('notebooks/Chapter3_CNN_Diabetes.ipynb', 'diabetes_prediction_dataset.csv', '../data/diabetes_prediction_dataset.csv')
fix_nb('notebooks/Chapter3_CNN_VietnamHousing.ipynb', 'vietnam_housing_dataset.csv', '../data/vietnam_housing_dataset.csv')
print("Done")
