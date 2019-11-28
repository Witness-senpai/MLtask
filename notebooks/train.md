# Импорты


```python
import gc

import catboost as cb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from malware_src.prepare_data import (dtypes,
                                    TARGET_COLUMN, 
                                    categorical_cols, 
                                    numerical_cols, 
                                    dataframe_to_pool)
from malware_src.visualization import plot_roc_curve
```

# Загрузка и разделение данных


```python
seed = 2019
val_part = 0.2
hold_part = 0.02
```


```python
%%time
data_path = 'data/filtered_train_data.csv'
data = pd.read_csv(data_path, dtype=dtypes)
```


```python
data.shape
```

## Подготовка категориальных данных для моделей


```python
# Выбираем OrdinalEncoder - как один из простейших и быстрых способов кодирования 
# категориальных данных. Например, OneHot кодирование мы не используем в целях 
# экономии ресурсов на вычисления и преобразования с работой полных данных. 
enc = OrdinalEncoder()
```


```python
# Пропуски как в категориальных, так и в числовых данных далее мы заменяем
# на специальное значение: -1. Этот способ самый простой и удобный, не требует 
# дополнительных вычислений, в отличии от, например, замены на самый частотный элемент
# для категориальных и замены на медиану или среднее в числовых данных.
# Убрать все NaN в категориальных столбцах и заменить их на '-1'.
# Так как тип данных - категориальный, NaN заполняются строкой
category_data = data.loc[:, categorical_cols]
for col in categorical_cols:
    try:
        category_data.loc[:, col] = category_data[col].fillna('-1')
    except ValueError:
        category_data[col] = category_data[col].cat.add_categories('-1')
        category_data.loc[:, col] = category_data[col].fillna('-1')
```


```python
category_data.info(null_counts=True)
```


```python
enc.fit(category_data)
```


```python
# Получаем nparray, заполненный числами, каждое из которых соответсвует
# своей категории в конкретном столбце.
filtered_category_data = enc.transform(category_data)
```


```python
numeric_data = data.loc[:, numerical_cols]
```


```python
filtered_numeric_data = numeric_data.to_numpy()
```


```python
# Убрать все NaN в числовых столбцах и заменить их на -1.
# Тип столбцов числовой, поэтому заменяется на -1, а не '-1'
for col in numerical_cols:
    try:
        numeric_data.loc[:, col] = numeric_data[col].fillna(-1)
    except ValueError:
        numeric_data[col] = numeric_data[col].cat.add_categories(-1)
        numeric_data.loc[:, col] = numeric_data[col].fillna(-1)
```


```python
full_data = np.concatenate((filtered_numeric_data, filtered_category_data), axis=1)
```


```python
full_data
```


```python
gc.collect()
```

### Разделение данных на тестовую и проверочную выборки


```python
train_data, val_data = train_test_split(full_data, test_size=val_part, random_state=seed)
```


```python
train_data = np.nan_to_num(train_data)
val_data = np.nan_to_num(val_data)
```


```python
np.argwhere(np.isnan(train_data))
```


```python
np.argwhere(np.isnan(val_data))
```

## Обучение моделей

### GaussianNB


```python
gauss_model = GaussianNB()
```


```python
%%time
gauss_model.fit(train_data[:,0:-1], train_data[:,-1])
```


```python
plt.figure(figsize=(10,10))
plot_roc_curve(true=val_data[:,-1],
               pred=gauss_model.predict_proba(val_data[:,0:-1])[:,1],
               name='ROC-кривая на валидационном датасете',
               p_label=1)
```

### BernoulliNB


```python
bernoulli_model = BernoulliNB()
```


```python
%%time
bernoulli_model.fit(train_data[:,0:-1], train_data[:,-1])
```


```python
plt.figure(figsize=(10,10))
plot_roc_curve(true=val_data[:,-1],
               pred=bernoulli_model.predict_proba(val_data[:,0:-1])[:,1],
               name='ROC-кривая на валидационном датасете',
               p_label=1)
```

### Keras sequential


```python
# Преобразование числового столбца в бинарный для работы с линейной моделью
binary_target = to_categorical(train_data[:, -1])
```


```python
%%time
seq_model = Sequential()

# Активационная функция softmax нужна для категориальных данных на выходе
seq_model.add(Dense(128, input_dim=len(train_data[0])-1, activation='sigmoid'))
seq_model.add(Dense(128, activation='sigmoid'))
seq_model.add(Dense(2, activation='softmax'))

# Так как таргет может быть либо 0 либо 1, то ошибку считаем по бинарной кроссэнтропии.
# Метрика точности так же задаётся для бинарного таргета 
seq_model.compile(loss='binary_crossentropy', 
                optimizer=Adam(), 
                metrics=['binary_accuracy'] 
                )

seq_model.fit(
            x=train_data[:, 0:-1],
            y=binary_target,
            epochs=50,
            batch_size=200000
            )
```


```python
plt.figure(figsize=(10,10))
plot_roc_curve(true=val_data[:, -1],
            pred=seq_model.predict_proba(val_data[:, 0:-1])[:,1],
            name='ROC-кривая на валидационном датасете',
            p_label=1)
```

### RandomForestClassifier


```python
%%time
rf_model = RandomForestClassifier(n_estimators=50)
rf_model.fit(train_data[:, 0:-1], binary_target)
```


```python
plt.figure(figsize=(10,10))
plot_roc_curve(true=val_data[:, -1],
            pred=rf_model.predict(val_data[:, 0:-1])[:, 1],
            name='ROC-кривая на валидационном датасете',
            p_label=1)
```

### Выводы

Простые модели наивного байеса показали себя чуть лучше, чем подкидывание монетки. При этом Гауссовская модель заметно отстаёт от Бернулли. Отсюда делается вывод, что исходные данные имеют мало общего с распредлением по Бернулли и тем более по Гауссу. 
Классическая линейная модель показала аналогичный результат с наивным байесом по Бернулли. Скорее всего, данные слишком сложные для предложенной структуры нейросети. Также, по логу keras видно, что ошибка за 2 первые эпохи спускается до конкретного числа и больше не может существенно сдвинуться с этой точки, это следствие попадания в локальный минимум. А лучше всего с существенным отрывом от других моделей себя показал случайный лес.

# Подготовка данных для Catboost

###  Обучение


```python
train_val_data, hold_data = train_test_split(data, test_size=hold_part, random_state=seed)
train_data, val_data = train_test_split(train_val_data, test_size=val_part, random_state=seed)
```


```python
print(data.shape[0], train_data.shape[0], val_data.shape[0], hold_data.shape[0])
```


```python
categorical_cols.remove(TARGET_COLUMN)
```


```python
train_pool = dataframe_to_pool(train_data, numerical_cols, categorical_cols, TARGET_COLUMN)
val_pool = dataframe_to_pool(val_data, numerical_cols, categorical_cols, TARGET_COLUMN)
```


```python
hold_data.to_csv('data/hold_data.csv', index=False)
```


```python
model = cb.CatBoostClassifier(iterations=100, learning_rate = 0.3)
```


```python
%%time
model.fit(train_pool, eval_set=val_pool, plot=True, verbose=True)
```


```python
model_path = 'models/model.cb'
model.save_model(model_path)
```


```python
feature_importance = model.get_feature_importance()
features = model.feature_names_

for feature_id in feature_importance.argsort()[::-1]:
    name = features[feature_id]
    importance = feature_importance[feature_id]
    print(f'{name:50}\t\t{importance}')
```

### Тест


```python
plt.figure(figsize=(10,10))
plot_roc_curve(true=val_pool.get_label(),
               pred=model.predict_proba(val_pool)[:,1],
               name='ROC-кривая на валидационном датасете',
               p_label='1')
```
