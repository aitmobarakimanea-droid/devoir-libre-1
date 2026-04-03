import pandas as pd

# charger dataset
df = pd.read_csv("titanic_survival.csv")

# afficher
print(df.head())

# valeurs manquantes
print(df.isnull().sum())

# supprimer embarked
df = df.dropna(subset=["embarked"])

# supprimer cabin
df = df.drop("cabin", axis=1)

# remplacer age
df["age"].fillna(df["age"].mean(), inplace=True)

# remplacer embarked
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)

# encoding
df = pd.get_dummies(df, columns=["embarked"])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])

# features
df = df[["pclass","sex","age","fare","survived"]]

# split
from sklearn.model_selection import train_test_split

X = df.drop("survived", axis=1)
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

print(X_train.shape)
print(X_test.shape)

# scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
