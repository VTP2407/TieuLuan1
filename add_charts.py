import nbformat as nbf

# 1. Update Diabetes Notebook
diabetes_nb = "notebooks/Chapter3_CNN_Diabetes.ipynb"
with open(diabetes_nb, 'r', encoding='utf-8') as f:
    nb_diab = nbf.read(f, as_version=4)

cm_cell = nbf.v4.new_code_cell("""# 7. Confusion Matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, labels_np) # Using NumPy scratch predictions

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (NumPy from Scratch)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
""")

nb_diab['cells'].append(cm_cell)
with open(diabetes_nb, 'w', encoding='utf-8') as f:
    nbf.write(nb_diab, f)


# 2. Update Vietnam Housing Notebook
housing_nb = "notebooks/Chapter3_CNN_VietnamHousing.ipynb"
with open(housing_nb, 'r', encoding='utf-8') as f:
    nb_hous = nbf.read(f, as_version=4)

scatter_cell = nbf.v4.new_code_cell("""# 7. Actual vs Predicted Scatter Plot
plt.figure(figsize=(8, 6))
# Using PyTorch predictions for the scatter plot as an example
plt.scatter(y_test, preds_test_pt, alpha=0.5, color='coral')

# Plot the ideal perfectly-predicted line
min_val = min(y_test.min(), preds_test_pt.min())
max_val = max(y_test.max(), preds_test_pt.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal Fit')

plt.title('Vietnam Housing: Actual vs Predicted Prices (PyTorch)')
plt.xlabel('Actual Price (Billion VND)')
plt.ylabel('Predicted Price (Billion VND)')
plt.legend()
plt.tight_layout()
plt.show()
""")

nb_hous['cells'].append(scatter_cell)
with open(housing_nb, 'w', encoding='utf-8') as f:
    nbf.write(nb_hous, f)

print("Added charts to both notebooks successfully.")
