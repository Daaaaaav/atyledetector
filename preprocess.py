import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('styles/train/_annotations.csv')
df.columns = df.columns.str.strip()

train_df = pd.DataFrame()
valid_df = pd.DataFrame()

for label in df['class'].unique():
    class_df = df[df['class'] == label]
    class_train, class_valid = train_test_split(class_df, test_size=0.2, stratify=class_df['class'])
    train_df = pd.concat([train_df, class_train])
    valid_df = pd.concat([valid_df, class_valid])

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(valid_df)}")

train_df.to_csv('styles/train/train_split.csv', index=False)
valid_df.to_csv('styles/valid/valid_split.csv', index=False)
