# Импорт библиотек


```python
import gc

import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt

from pprint import pprint

plt.style.use('seaborn-whitegrid')
sns.set(style="ticks", color_codes=True)
```

# Анализ данных

Источник данных: https://www.kaggle.com/c/microsoft-malware-prediction/data


```python
train_data_path = '../data/train.csv'
```


```python
# Заранее предопределённые типы, чтобы при загрузке всех данных с помощью pandas
# расходовалось намного меньше памяти. 
dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float32',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int16',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float64', 
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float32', 
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float32', 
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float64', 
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float64', 
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float32', 
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float32', 
        'Census_InternalPrimaryDisplayResolutionVertical':      'float32', 
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float64', 
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }
```


```python
train_data = pd.read_csv(train_data_path, dtype=dtypes)
```


```python
train_data.info()
```


```python
# Удостовериваемся, что импорт прошёл верно, просмотрев все столбцы в первых 5 строчках
pd.set_option('display.max_columns', len(train_data.columns))  
train_data.head()
```


```python
cols = train_data.columns.to_list()
```


```python
train_data.describe().T
```


```python
# True numerical columns according the description of the data
numerical_cols = [
    'Census_ProcessorCoreCount',
    'Census_PrimaryDiskTotalCapacity',
    'Census_SystemVolumeTotalCapacity',
    'Census_TotalPhysicalRAM',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches',
    'Census_InternalPrimaryDisplayResolutionHorizontal',
    'Census_InternalPrimaryDisplayResolutionVertical',
    'Census_InternalBatteryNumberOfCharges'
]
categorical_cols = []

stat_cols = [('name', 'unique values', 'part of most popular')]

# Analisis each column and grouping by catecorical features
for col in cols[1:]:
    col_stat = train_data[col].value_counts()
    unique_values = len(col_stat)
    part_most_popular_val = col_stat.iloc[0] / col_stat.sum()
    stat_cols.append((col, unique_values, part_most_popular_val))

# Getting max width of each column for next printing
max_widths = []
stat_cols_transpose = np.asarray(stat_cols).transpose()
for row in stat_cols_transpose:
    max_widths.append(max( len(str(el)) for el in row ))

# Formatted printing
for row in stat_cols:
    formated_row = ''.join([str(el).ljust(max_widths[n]+2) for n,el in enumerate(row)])
    print(formated_row)
```

Заметим, что существуют столбцы, с очень большой долей самого популярного элемента. При этом, количество уникальных элементов в них очень малое. Напрашивается вывод, что такие столбцы вряд ли смогут повлиять на результирующий столбец. Предположим, что столбцы с долей самого популярного элемента больше 98% не повлияют на результат. В следующей ячейке наглядно показаны такие столбцы.


```python
MOST_PART = 0.98
useless_cols = []

# Пропуск первого элемента с названиями столбцов форматированого вывода
for stat in stat_cols[1:]:
    # 2 элемент кортежа stat показывает часть самого популярного значения в столбце
    if (stat[2] > MOST_PART):
        pprint(stat)
        useless_cols.append(stat[0])
```


```python
categorical_cols = [col for col in cols[1:] if col not in useless_cols + numerical_cols]
```


```python
pprint(categorical_cols)
```


```python
pprint(numerical_cols)
```

# Сохраняем данные только с необходимыми столбцами


```python
train_data.drop(columns=useless_cols, inplace=True)
```


```python
train_data.to_csv('../data/filtered_train_data.csv', index=False)
```

# Небольшая визуализация


```python
_ = train_data[numerical_cols].hist(figsize=(35,35))
```


```python
_ = train_data[categorical_cols].hist(figsize=(35,35))
```
