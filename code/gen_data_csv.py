import pandas as pd
import glob
from sklearn.model_selection import train_test_split

raw_data_df = pd.read_csv('styles.csv')
# print(raw_data_df.head())

labels_df = raw_data_df[['id', 'subCategory', 'articleType', 'baseColour']]
labels_df = labels_df.dropna()
labels_df = labels_df[(labels_df['subCategory'] =='Topwear') | (labels_df['subCategory'] == 'Bottomwear') | (labels_df['subCategory'] == 'Dress')]
image_base_dir = 'crop_images'
image_fnames = [f for f in glob.glob(image_base_dir + "**/*.jpg", recursive=True)]
print(11111111111, len(image_fnames))
image_ids = [int(f.split('/')[-1][:-4]) for f in image_fnames]
image_ids_df = pd.DataFrame({'id':image_ids,'fname':image_fnames}, index=None)
labels_df = labels_df.merge(image_ids_df, how='inner', on='id')
labels_df.reset_index(drop='index', inplace=True)

transform_dict = {
    # 'MenShirt': ['Tshirts', 'Shirts'],
    'WomenShirt': ['Kurtas', 'Tops', 'Kurtis', 'Tunics'],
    'Covering': ['Sweatshirts', 'Sweaters', 'Jackets'],
    'LongPants': ['Jeans', 'Trousers', 'Track Pants', 'Leggings'],
    'ShortPants': ['Shorts', 'Capris'],
    'Dress': ['Skirts', 'Dresses']
}
new_df= labels_df[labels_df.groupby('articleType').articleType.transform('count')>100].copy() 
for key in transform_dict:
  for i in transform_dict[key]:
    new_df.loc[new_df['articleType'] == i, 'articleType'] = key

new_df = new_df[new_df['articleType'] != 'Dupatta']
new_df= new_df[new_df.groupby('baseColour').baseColour.transform('count')>50].copy() 
# print(new_df.articleType.value_counts())
new_df = new_df[['id', 'articleType', 'baseColour', 'fname']]
# print(new_df)

X = new_df['fname']
y = new_df.drop(columns=['id', 'fname'])

# print(X.head())
# print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=new_df['baseColour'], random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train['baseColour'], random_state=42)
# print(y_train.articleType.value_counts())
# print(y_train.baseColour.value_counts())

X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print(X_train)

new_folder = 'datacsv/'
X_train.to_csv(new_folder + 'X_train.csv')
X_test.to_csv(new_folder + 'X_test.csv')
X_val.to_csv(new_folder + 'X_val.csv')
y_train.to_csv(new_folder + 'y_train.csv')
y_test.to_csv(new_folder + 'y_test.csv')
y_val.to_csv(new_folder + 'y_val.csv')