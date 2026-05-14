import pandas as pd
def find_s(path):
    df = pd.read_csv(path); print("Training data:\n", df)
    h = ['?'] * (len(df.columns)-1)
    for _, row in df[df[df.columns[-1]] == 'Yes'].iterrows():
        for i, v in enumerate(row[:-1]):
            h[i] = v if h[i] == '?' or h[i] == v else '?'
    return h
path = "training_data.csv"
print("\nThe final hypothesis is:", find_s(path))
