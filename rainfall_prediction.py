import pandas as pd  

df = pd.read_csv("AUSweather_data.csv")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nMissing Values: ")
print(df.isnull().sum())

df = df.drop(
    columns=["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"]
)

print("\nShape after dropping columns: ")
print(df.shape)

print("\nColumns after dropping: ")
print(df.columns)


df = df.dropna(subset=["RainTomorrow"])

print(df.shape)

print(df.isnull().sum())

num_cols = df.select_dtypes(include=["float64", "int64"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())


print(df.isnull().sum())

cat_cols = df.select_dtypes(include=["object"]).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print(df.isnull().sum())

print(df["RainTomorrow"].value_counts())
print(df["RainToday"].value_counts())

df["RainTomorrow"] = df["RainTomorrow"].map({
    "No": 0,
    "Yes": 1
})

df["RainToday"] = df["RainToday"].map({
    "No": 0,
    "Yes": 1
})

print(df["RainTomorrow"].unique())
print(df["RainToday"].unique())

print(df[["Location", "WindGustDir", "WindDir9am", "WindDir3pm"]].head())

df =pd.get_dummies(df, columns=["Location", "WindGustDir", "WindDir9am", "WindDir3pm"], drop_first=True)

print(df.shape)
print(df.head())

df = df.drop("Date", axis=1)

print(df["RainTomorrow"].unique())


X = df.drop("RainTomorrow", axis=1)
y = df["RainTomorrow"]

print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: ", accuracy)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
