# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Detect File Encoding: Use chardet to determine the dataset's encoding.
2.Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3.Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4.Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5.Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6.Train SVM Model: Fit an SVC model on the training data.
7.Predict Labels: Predict test labels using the trained SVM model.
8.Evaluate Model: Calculate and display accuracy with metrics.accuracy_scor

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: AADHITHYAA L
RegisterNumber: 212224220003
print("AADHITHYAA L")
print("212224220003")
import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect (rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv('spam.csv', encoding='Windows-1252')
data.head()
print("AADHITYAA L")
print("212224220003")
data.info()
data.isnull().sum()
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train
x_test
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
x_train
x_test
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


```

## Output:
RESULT:

<img width="980" height="97" alt="image" src="https://github.com/user-attachments/assets/b9a354bd-eb91-400a-9faa-e834960701af" />

HEAD:

<img width="1076" height="221" alt="image" src="https://github.com/user-attachments/assets/baa82b9e-23dc-4596-b414-8d2ddae6c19e" />

INFO:

<img width="740" height="317" alt="image" src="https://github.com/user-attachments/assets/d66de79f-f65d-48f4-95f7-ccf62c6bb881" />


DATA.ISNULL,SUM()

<img width="637" height="147" alt="image" src="https://github.com/user-attachments/assets/00312424-dc6c-4403-b4f9-ba026f75c511" />

X_TRAIN:

<img width="1381" height="160" alt="image" src="https://github.com/user-attachments/assets/2d2f146c-e14d-48f1-baaa-7ee742b8e4b6" />


X_TEST:

<img width="1016" height="52" alt="image" src="https://github.com/user-attachments/assets/c3047fc8-a9f1-4a07-82fe-bba5194877e2" />

Y_PRED:

<img width="913" height="48" alt="image" src="https://github.com/user-attachments/assets/538815c9-74ab-4518-87d1-20543ab2e0fe" />

ACCURACY:

<img width="718" height="52" alt="image" src="https://github.com/user-attachments/assets/c1b404cd-742d-4cef-bbf1-ffcb9952f8d5" />







## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
