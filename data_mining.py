from re import L
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv('diabetes.csv')


print(df)
#print(df.isnull().sum()) #sıfır bulunduran veriler
#print(df.eq(0).sum()) #hangi veride kaç tane sıfır var
#print(df.shape) #768 hasta 9 özellik
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0,np.NaN) #0 olmaması gereken eksik veriler
#eksik değerlerin doldurulması
df.fillna(df.mean(), inplace=True) #ortalama ile doldurma
#print(df.eq(0).sum()) #glikozda 0 değeri kalmadı
#KORELASYON ANALİZİ -- İKİ DEĞİKEN ARASI İLİŞKİ ÖLÇÜMÜ -- değişkenler çıktıyı ne kadar etkiler
"""
En fazla alakalı 4 değer

sns.heatmap(df.corr().nlargest(4,'Outcome'))
plt.tight_layout()
plt.xlabel('Corr')
plt.show()
"""
""" Buraya kadar olan kısım veri önişlemesidir.
Kullanılcak algoritmalar 
    1.lojistik regrasyon
    2.destek vektör regrasyonu
"""
#algoritmalar oluşturulması
#elimizdeki 100 veriden her 10 gözlem değerinden 1 öğrenme gerçekleşir.
"""
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
X = df[['Glucose', 'BMI', 'Age']]
y = df.iloc[:, 8]
"""
#print(X)
#print(y)
#LOG REG
"""
log_reg = linear_model.LogisticRegression()
log_reg_score = cross_val_score(log_reg,X,y,cv= 10, scoring='accuracy').mean()
print("lojistik regresyon sonucu: %",log_reg_score*100) #500 tane 0'ın %76 sını doğru bilmişiz.
results = []
results.append(log_reg_score)
from sklearn import svm
linear_svm = svm.SVC(kernel='linear')
linear_svm_score = cross_val_score(linear_svm,X,y,cv=10,scoring='accuracy').mean()
print("destek vektör regrasyon sonucu: %", linear_svm_score*100) #iki algoritmada lojistik regresyonu tercih ettik.

results.append(linear_svm_score)
#print(results)
import pickle
filename = 'diabets.sav'
log_reg.fit(X,y)
pickle.dump(log_reg, open(filename,'wb')) #okunamıyor, daha sonra geri de çağrılabilir. iyi öğrenmelerin kaybolmaması için 
loaded_model = pickle.load(open(filename,'rb')) #modelin çağırılması
Glucose = 100
BMI = 40
Age = 40
prediction = loaded_model.predict([[Glucose, BMI, Age]]) # gelen kişi hasta mı?
print(prediction)
"""
