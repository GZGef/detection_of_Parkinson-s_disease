"""Импортируем библиотеки"""

from requests import get
import pandas as pd
import numpy as np

from plotly import graph_objects as go
from matplotlib import pyplot as plt
import seaborn as sns

import tempfile
import os
from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import xgboost as XGB

"""### Загружаем датасет"""

# Создаем запрос и получаем с него файл
response = get("https://storage.yandexcloud.net/academy.ai/practica/parkinsons.data")

# Открываем в режиме записи байтов
with open("parkinsons.data", 'wb') as file:
    file.write(response.content)

# Создаем объект DataFrame
df = pd.read_csv("parkinsons.data", sep=',')
df.info()

df.describe()

df.head()

# Получим последние пять строк набора данных
df.tail()

"""### Проверка на наличие нулевых или пропущенных значений"""

df.isnull().sum()

"""### Построим диаграмму рассеивания"""

sns.pairplot(data = df[df.columns[0:24]])
plt.show()

# Получим массив признаков и статусов на наличие болезни
all_features = df.loc[:, df.columns != 'status'].values[:, 1:] # Признаки
out_come = df.loc[:, 'status'].values # Статусы

print(
f'''
Количество людей с БП: {out_come[out_come == 1].shape[0]}
Здоровых людей: {out_come[out_come == 0].shape[0]}
'''
)

"""### Масштабируем данные"""

scaler = MinMaxScaler((-1, 1)) # Указываем минимальное и максимальное значения для масштабирования
x = scaler.fit_transform(all_features)
y = out_come

print(f"x: {x}")

"""### Разделение данных для обучения и тестирования"""

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size = 0.2, # Доля данных, которая будет выделена для тестовой выборки
    random_state = 3 # Начальное состояние генератора случайных чисел
)

print(f'Размерность x_train {x_train.shape}')
print(f'Размерность y_train {y_train.shape}')

"""## Инициализируем эти классификаторы:
*   Support Vector Classification
*   XGBClassifier
*   KNeighborsClassifier
*   DecisionTreeClassifier
*   RandomForestClassifier
*   Regression classifier
*   GaussianNB

И обучаем на основе каждого классификатора отдельную модель
"""

svm = SVC(
    kernel = 'rbf',
    random_state = 0,
    gamma = .10,
    C = 1.0
)

svm.fit(x_train, y_train)

xgb_clf = XGB.XGBClassifier()
xgb_clf = xgb_clf.fit(x_train, y_train)

knn = KNeighborsClassifier(
    n_neighbors = 5,
    metric = 'minkowski',
    p = 2
)

knn.fit(x_train, y_train)

decision_tree = tree.DecisionTreeClassifier(criterion = 'gini')
decision_tree.fit(x_train, y_train)

random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train)

lg = LogisticRegression(solver = 'lbfgs')
lg.fit(x_train, y_train)

nb = GaussianNB()
nb.fit(x_train, y_train)

print('Точность на тренировочных и тестовых данных')
print('')
print('')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('')
print('Точность классификатора SVM по обучающим данным равна: {:.2f}'.format(svm.score(x_train, y_train)*100))
print('Точность классификатора SVM по тестовым данным составляет: {:.2f}'.format(svm.score(x_test, y_test)*100))
print('')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('')
print('Точность классификатора XGBoost по обучающим данным равна: {:.2f}'.format(xgb_clf.score(x_train, y_train)*100))
print('Точность классификатора XGBoost по тестовым данным составляет: {:.2f}'.format(xgb_clf.score(x_test, y_test)*100))
print('')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('')
print('Точность классификатора Knn по обучающим данным равна: {:.2f}'.format(knn.score(x_train, y_train)*100))
print('Точность классификатора Knn по тестовым данным составляет: {:.2f}'.format(knn.score(x_test, y_test)*100))
print('')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('')
print('Точность классификатора дерева решений по обучающим данным равна: {:.2f}'.format(decision_tree.score(x_train, y_train)*100))
print('Точность классификатора дерева решений по тестовым данным составляет: {:.2f}'.format(decision_tree.score(x_test, y_test)*100))
print('')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('')
print('Точность классификатора случайного леса по обучающим данным равна: {:.2f}'.format(random_forest.score(x_train, y_train)*100))
print('Точность классификатора случайного леса по тестовым данным составляет: {:.2f}'.format(random_forest.score(x_test, y_test)*100))
print('')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('')
print('Точность классификатора логистической регрессии по обучающим данным равна: {:.2f}'.format(lg.score(x_train, y_train)*100))
print('Точность классификатора логистической регрессии по тестовым данным составляет: {:.2f}'.format(lg.score(x_test, y_test)*100))
print('')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('')
print('Точность классификатора GaussianNB для обучающих данных равна: {:.2f}'.format(nb.score(x_train, y_train)*100))
print('Точность классификатора GaussianNB по тестовым данным составляет: {:.2f}'.format(nb.score(x_test, y_test)*100))