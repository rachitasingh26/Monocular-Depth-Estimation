import pandas as pd

df = pd.read_csv('/projectnb/dl4ds/materials/datasets/monocular-depth-estimation/nyuv2/nyu_data/data/nyu2_train.csv', header=None)
# print(df[0][0])
for index, row in df.iterrows():
    rgb = row[0]
    ground_truth = row[1]
    print(rgb, ground_truth)