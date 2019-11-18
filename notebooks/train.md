# Импорты


```python
import pickle
import gc

import catboost as cb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from malware_src.prepare_data import dataframe_to_pool
from malware_src.visualization import plot_roc_curve
```

# Загрузка и разделение данных


```python
dtypes = {
    'EngineVersion':                                     'category',
    'AppVersion':                                        'category',
    'AvSigVersion':                                      'category',
    'RtpStateBitfield':                                  'category',
    'DefaultBrowsersIdentifier':                         'category',
    'AVProductStatesIdentifier':                         'category',
    'AVProductsInstalled':                               'category',
    'AVProductsEnabled':                                 'category',
    'CountryIdentifier':                                 'category',
    'CityIdentifier':                                    'category',
    'OrganizationIdentifier':                            'category',
    'GeoNameIdentifier':                                 'category',
    'LocaleEnglishNameIdentifier':                       'category',
    'Platform':                                          'category',
    'Processor':                                         'category',
    'OsVer':                                             'category',
    'OsBuild':                                           'category',
    'OsSuite':                                           'category',
    'OsPlatformSubRelease':                              'category',
    'OsBuildLab':                                        'category',
    'SkuEdition':                                        'category',
    'IsProtected':                                       'category',
    'IeVerIdentifier':                                   'category',
    'SmartScreen':                                       'category',
    'Firewall':                                          'category',
    'Census_MDC2FormFactor':                             'category',
    'Census_OEMNameIdentifier':                          'category',
    'Census_OEMModelIdentifier':                         'category',
    'Census_ProcessorManufacturerIdentifier':            'category',
    'Census_ProcessorModelIdentifier':                   'category',
    'Census_ProcessorClass':                             'category',
    'Census_PrimaryDiskTypeName':                        'category',
    'Census_HasOpticalDiskDrive':                        'category',
    'Census_ChassisTypeName':                            'category',
    'Census_PowerPlatformRoleName':                      'category',
    'Census_InternalBatteryType':                        'category',
    'Census_OSVersion':                                  'category',
    'Census_OSArchitecture':                             'category',
    'Census_OSBranch':                                   'category',
    'Census_OSBuildNumber':                              'category',
    'Census_OSBuildRevision':                            'category',
    'Census_OSEdition':                                  'category',
    'Census_OSSkuName':                                  'category',
    'Census_OSInstallTypeName':                          'category',
    'Census_OSInstallLanguageIdentifier':                'category',
    'Census_OSUILocaleIdentifier':                       'category',
    'Census_OSWUAutoUpdateOptionsName':                  'category',
    'Census_GenuineStateName':                           'category',
    'Census_ActivationChannel':                          'category',
    'Census_FlightRing':                                 'category',
    'Census_FirmwareManufacturerIdentifier':             'category',
    'Census_FirmwareVersionIdentifier':                  'category',
    'Census_IsSecureBootEnabled':                        'category',
    'Census_IsTouchEnabled':                             'category',
    'Census_IsPenCapable':                               'category',
    'Census_IsAlwaysOnAlwaysConnectedCapable':           'category',
    'Wdft_IsGamer':                                      'category',
    'Wdft_RegionIdentifier':                             'category',
    'HasDetections':                                     'category',
    'Census_ProcessorCoreCount':                         'float16',
    'Census_PrimaryDiskTotalCapacity':                   'float64',
    'Census_SystemVolumeTotalCapacity':                  'float64',
    'Census_TotalPhysicalRAM':                           'float32',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches': 'float32',
    'Census_InternalPrimaryDisplayResolutionHorizontal': 'float32',
    'Census_InternalPrimaryDisplayResolutionVertical':   'float32',
    'Census_InternalBatteryNumberOfCharges':             'float64'
}

TARGET_COLUMN = 'HasDetections'
```


```python
categorical_cols = [el for el in dtypes if dtypes[el] == 'category']
numerical_cols = [el for el in dtypes if dtypes[el] != 'category']
```


```python
seed = 2019
val_part = 0.2
hold_part = 0.02
```


```python
%%time
data_path = '../data/filtered_train_data.csv'
data = pd.read_csv(data_path, dtype=dtypes)
```


```python
data.shape[0]
```

## Подготовка данных для моделей из naive bayes


```python
enc = OrdinalEncoder()
```


```python
data.info()
```


```python
# Убрать все NaN в категориальных столбцах
category_data = pd.DataFrame(data, columns=categorical_cols)
for col in categorical_cols:
    try:
        category_data.loc[:, col] = category_data[col].fillna('-1')
    except:
        category_data[col] = category_data[col].cat.add_categories('-1')
        category_data.loc[:, col] = category_data[col].fillna('-1')
```


```python
enc.fit(category_data)
```


```python
category_data.info()
```


```python
filtered_category_data = enc.transform(category_data)
```


```python
numeric_data = pd.DataFrame(data, columns=numerical_cols)
```


```python
filtered_numeric_data = numeric_data.to_numpy()
```


```python
# Убрать все NaN в числовых столбцах
for col in numerical_cols:
    try:
        numeric_data.loc[:, col] = numeric_data[col].fillna(-1)
    except:
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
train_data
```


```python
val_data
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
               name='ROC-кривая на валидационном датасете')
```


```python
gc.collect()
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
               name='ROC-кривая на валидационном датасете')
```

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
try:
    categorical_cols.remove(TARGET_COLUMN)
except Exception as ex:
    print(ex)
train_pool = dataframe_to_pool(train_data, numerical_cols, categorical_cols, TARGET_COLUMN)
val_pool = dataframe_to_pool(val_data, numerical_cols, categorical_cols, TARGET_COLUMN)
```


```python
hold_data.to_csv('../data/hold_data.csv', index=False)
```


```python
gc.collect()
```


```python
model = cb.CatBoostClassifier(iterations=100, learning_rate = 0.3)
```


```python
%%time
model.fit(train_pool, eval_set=val_pool, plot=True, verbose=True)
```


```python
model_path = '../models/model.cb'
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
def plot_roc_curve(true,
                   pred,
                   name,
                   weights=None,
                   label='',
                   color='darkorange',
                   ax=None,
                   p_label='1',
):
    if ax is None:
        ax = plt.gca()
    fpr, tpr, thr = roc_curve(true, pred, sample_weight=weights, pos_label=p_label)
    lw = 2
    ax.plot(fpr,
            tpr,
            color=color,
            lw=lw,
            label=('ROC {} curve (area = {:0.3f})'
                   .format(label, roc_auc_score(true,
                                                pred,
                                                sample_weight=weights))))
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(name)
    ax.legend(loc="lower right")
```


```python
model.load_model('../models/model.cb')
```


```python
val_data = pd.read_csv('../data/hold_data.csv', dtype=dtypes)
```


```python
val_pool = dataframe_to_pool(val_data, numerical_cols, categorical_cols, TARGET_COLUMN)
```


```python
plt.figure(figsize=(10,10))
plot_roc_curve(true=val_pool.get_label(),
               pred=model.predict_proba(val_pool)[:,1],
               name='ROC-кривая на валидационном датасете',
               p_label='1')
```
