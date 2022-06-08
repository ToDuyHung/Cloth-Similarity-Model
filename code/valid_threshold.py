import pandas as pd

df = pd.read_csv('/AIHCM/ComputerVision/hungtd/fashion-dataset/datacsv/distance.csv')
threshold = 4.1
print(len(df))
print(len(df[df['positive'] < threshold]))
print(len(df[df['negative'] >= threshold]))