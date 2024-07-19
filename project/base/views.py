from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../Notebooks/titanic.csv')
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns='Cabin', inplace=True)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])
df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
#print(f'Accuracy: {accuracy:.2f}')
survival_probabilities = model.predict_proba(X_test)[:, 1]

def predict(Pclass,sex,age,sibsp,parch,fare,embarked):
    input_data=np.array([[Pclass,sex,age,sibsp,parch,fare,embarked]])
    scaled=scaler.transform(input_data)
    return round(model.predict_proba(scaled)[:,1][0],2)

# Create your views here.
def home(request):
    return render(request,'predict.html')

@csrf_exempt
def predict_survival(request):
    if request.method=='POST':
        data=request.POST
        #print(data)
        feautures=[
            int(data['Pclass']),
            int(data['Sex']),
            int(data['Age']),
            int(data['SibSp']),
            int(data['Parch']),
            float(data['Fare']),
            int(data['Embarked'])
        ]
        #print(feautures)
        survival_probability=predict(*feautures)*100
        context={
            #'prediction':int(prediction[0]),
            'survival_probability':survival_probability
        }
        #print(survival_probability)
        return render(request,'result.html',context)
    return JsonResponse({'error':'invalid request method'}, status=400)
