import os
import pandas as pd
from sklearn.datasets import load_iris

def download_data(output_path):
    
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    os.makedirs('data', exist_ok=True)
    df.to_csv(output_path, index=False)
    print('The dataset has been loaded.')

# Загружаем и сохраняем данные
download_data('data/dataset.csv')