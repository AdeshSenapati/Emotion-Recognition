import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics
import pickle

df = pd.read_csv('coords.csv')
# print(df[df['class']=='Sad'])
X = df.drop('class', axis=1)  # features
y = df['class']  # target value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
# test size 0.3 means we are taking testing partition of 30% random state ensures we get similar results
# whenever we run it
pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}  # consider these pipelines as 4 seperate individual machine learning pipeline models
# so acc. to the code first the model goes through standard scaler to bring data to a level basis
# and then it will go to the main models.

fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)  # fit method is complicated so think of it as training method basically
    # means train
    fit_models[algo] = model

print("Training of 4 models done... checking accuracy next....")

ls = {}
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    ls[algo] = accuracy_score(y_test, yhat)
    print(algo, accuracy_score(y_test, yhat))

max_key = max(ls, key=ls.get)
print(f"{max_key} was chosen as the best case prediction model...")

with open('Emotion_detection_model.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)

print("Done training and creating prediction model and ready for use.... ")
