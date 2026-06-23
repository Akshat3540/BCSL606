import pandas as pd
def find_s():
    df = pd.read_csv("training_data.csv")
    print("Training data:\n", df)
    h = [None] * (len(df.columns)-1)
    for _, row in df[df[df.columns[-1]] == 'Yes'].iterrows():
        for i, v in enumerate(row[:-1]):
            h[i] = v if h[i] is None or h[i] == v else '?'
    return h
print("\nThe final hypothesis is:", find_s())