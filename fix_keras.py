import nbformat

def fix_keras_warning(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    changed = False
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if 'SimpleRNN(16, input_shape=(window_size, 1))' in cell.source:
                cell.source = cell.source.replace(
                    'SimpleRNN(16, input_shape=(window_size, 1))', 
                    'tf.keras.Input(shape=(window_size, 1)),\n    SimpleRNN(16)'
                )
                changed = True
                
    if changed:
        with open(file_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f'Fixed warning in {file_path}')

fix_keras_warning('notebooks/Chapter4_RNN_Amazon.ipynb')
fix_keras_warning('notebooks/Chapter4_RNN_Gold.ipynb')
