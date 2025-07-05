import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv(r'C:\Users\SANJE\OneDrive\Pictures\astro_flask_app\star_classification.csv')

print(df.head())

X = df[['u', 'g', 'r', 'i', 'z', 'redshift']] 
y = df['class']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'astronomy_model.pkl')
print("Model saved as 'astronomy_model.pkl'")
