# Импорты


```python
import pickle
import gc

import catboost as cb
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
#from malware_src.prepare_data import dataframe_to_pool
#from malware_src.visualization import plot_roc_curve
#from malware_src.model import load_model, apply_model_to_json
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
```


```python
categorical = [el for el in dtypes if dtypes[el] == 'category']
numerical = [el for el in dtypes if dtypes[el] != 'category']
```


```python
seed = 2019
val_part = 0.2
```


```python
%%time
data_path = '../data/filtered_train_data.csv'
data = pd.read_csv(data_path, dtype=dtypes)
train_data, val_data = train_test_split(data, test_size=val_part, random_state=seed)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <timed exec> in <module>
    

    TypeError: dataframe_to_pool() missing 3 required positional arguments: 'numeric_columns', 'categorical_columns', and 'target_column'



```python
train_data.shape[0]
```




    1000000




```python
train_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000000 entries, 0 to 999999
    Data columns (total 67 columns):
    EngineVersion                                        1000000 non-null category
    AppVersion                                           1000000 non-null category
    AvSigVersion                                         1000000 non-null category
    RtpStateBitfield                                     996334 non-null category
    DefaultBrowsersIdentifier                            48303 non-null category
    AVProductStatesIdentifier                            995907 non-null category
    AVProductsInstalled                                  995907 non-null category
    AVProductsEnabled                                    995907 non-null category
    CountryIdentifier                                    1000000 non-null category
    CityIdentifier                                       963487 non-null category
    OrganizationIdentifier                               691064 non-null category
    GeoNameIdentifier                                    999981 non-null category
    LocaleEnglishNameIdentifier                          1000000 non-null category
    Platform                                             1000000 non-null category
    Processor                                            1000000 non-null category
    OsVer                                                1000000 non-null category
    OsBuild                                              1000000 non-null category
    OsSuite                                              1000000 non-null category
    OsPlatformSubRelease                                 1000000 non-null category
    OsBuildLab                                           999997 non-null category
    SkuEdition                                           1000000 non-null category
    IsProtected                                          995924 non-null category
    IeVerIdentifier                                      993347 non-null category
    SmartScreen                                          644223 non-null category
    Firewall                                             989660 non-null category
    Census_MDC2FormFactor                                1000000 non-null category
    Census_OEMNameIdentifier                             989473 non-null category
    Census_OEMModelIdentifier                            988687 non-null category
    Census_ProcessorCoreCount                            995368 non-null float16
    Census_ProcessorManufacturerIdentifier               995367 non-null category
    Census_ProcessorModelIdentifier                      995362 non-null category
    Census_ProcessorClass                                4125 non-null category
    Census_PrimaryDiskTotalCapacity                      994118 non-null float64
    Census_PrimaryDiskTypeName                           998544 non-null category
    Census_SystemVolumeTotalCapacity                     994119 non-null float64
    Census_HasOpticalDiskDrive                           1000000 non-null category
    Census_TotalPhysicalRAM                              991004 non-null float32
    Census_ChassisTypeName                               999939 non-null category
    Census_InternalPrimaryDiagonalDisplaySizeInInches    994619 non-null float32
    Census_InternalPrimaryDisplayResolutionHorizontal    994628 non-null float32
    Census_InternalPrimaryDisplayResolutionVertical      994628 non-null float32
    Census_PowerPlatformRoleName                         999994 non-null category
    Census_InternalBatteryType                           290170 non-null category
    Census_InternalBatteryNumberOfCharges                969924 non-null float64
    Census_OSVersion                                     1000000 non-null category
    Census_OSArchitecture                                1000000 non-null category
    Census_OSBranch                                      1000000 non-null category
    Census_OSBuildNumber                                 1000000 non-null category
    Census_OSBuildRevision                               1000000 non-null category
    Census_OSEdition                                     1000000 non-null category
    Census_OSSkuName                                     1000000 non-null category
    Census_OSInstallTypeName                             1000000 non-null category
    Census_OSInstallLanguageIdentifier                   993300 non-null category
    Census_OSUILocaleIdentifier                          1000000 non-null category
    Census_OSWUAutoUpdateOptionsName                     1000000 non-null category
    Census_GenuineStateName                              1000000 non-null category
    Census_ActivationChannel                             1000000 non-null category
    Census_FlightRing                                    1000000 non-null category
    Census_FirmwareManufacturerIdentifier                979547 non-null category
    Census_FirmwareVersionIdentifier                     982112 non-null category
    Census_IsSecureBootEnabled                           1000000 non-null category
    Census_IsTouchEnabled                                1000000 non-null category
    Census_IsPenCapable                                  1000000 non-null category
    Census_IsAlwaysOnAlwaysConnectedCapable              992020 non-null category
    Wdft_IsGamer                                         965907 non-null category
    Wdft_RegionIdentifier                                965907 non-null category
    HasDetections                                        1000000 non-null int64
    dtypes: category(58), float16(1), float32(4), float64(3), int64(1)
    memory usage: 130.5 MB
    

# Catboost

###  Обучение


```python
def dataframe_to_pool(df,
               numeric_columns,
               categorical_columns,
               target_column,
               ):
    columns = numeric_columns + categorical_columns
    target = None
    if target_column in df.columns:
        target = df[target_column]
    df = df[columns].copy()
    for col in categorical_columns:
        df[col] = df[col].cat.add_categories(-1)
        df.loc[:, col] = df[col].fillna(-1)
    return cb.Pool(
            df,
            label=target,
            cat_features=categorical_columns,
        )


def json_to_pool(json_data,
            numeric_columns,
            categorical_columns,
            target_column
            ):
    df = pd.DataFrame(json_data, index=[0])
    return dataframe_to_pool(df,
                        numeric_columns,
                        categorical_columns,
                        target_column)
```


```python
model = cb.CatBoostClassifier(iterations=100)
train_pool = dataframe_to_pool(train_data, numerical, categorical, 'HasDetections')
val_pool = dataframe_to_pool(val_data, numerical, categorical, 'HasDetections')
```


```python
gc.collect()
```




    20




```python
%%time
#model.fit(train_pool, eval_set=val_pool, verbose=True)
model.fit(train_pool, eval_set=val_pool, verbose=True)
```

    Learning rate set to 0.5
    0:	learn: 0.6540201	test: 0.6540800	best: 0.6540800 (0)	total: 25.3s	remaining: 41m 47s
    1:	learn: 0.6391347	test: 0.6391074	best: 0.6391074 (1)	total: 40.1s	remaining: 32m 46s
    2:	learn: 0.6337850	test: 0.6336888	best: 0.6336888 (2)	total: 55s	remaining: 29m 37s
    3:	learn: 0.6308730	test: 0.6307271	best: 0.6307271 (3)	total: 1m 3s	remaining: 25m 29s
    4:	learn: 0.6291877	test: 0.6290695	best: 0.6290695 (4)	total: 1m 12s	remaining: 22m 59s
    5:	learn: 0.6276804	test: 0.6273907	best: 0.6273907 (5)	total: 1m 21s	remaining: 21m 16s
    6:	learn: 0.6262344	test: 0.6258957	best: 0.6258957 (6)	total: 1m 30s	remaining: 19m 58s
    7:	learn: 0.6252441	test: 0.6249058	best: 0.6249058 (7)	total: 1m 39s	remaining: 19m 1s
    8:	learn: 0.6245115	test: 0.6241608	best: 0.6241608 (8)	total: 1m 48s	remaining: 18m 21s
    9:	learn: 0.6238581	test: 0.6235635	best: 0.6235635 (9)	total: 1m 59s	remaining: 17m 54s
    10:	learn: 0.6227701	test: 0.6224830	best: 0.6224830 (10)	total: 2m 8s	remaining: 17m 23s
    11:	learn: 0.6222859	test: 0.6219594	best: 0.6219594 (11)	total: 2m 18s	remaining: 16m 54s
    12:	learn: 0.6217746	test: 0.6214381	best: 0.6214381 (12)	total: 2m 27s	remaining: 16m 30s
    13:	learn: 0.6211670	test: 0.6208183	best: 0.6208183 (13)	total: 2m 37s	remaining: 16m 10s
    14:	learn: 0.6206721	test: 0.6203272	best: 0.6203272 (14)	total: 2m 47s	remaining: 15m 47s
    15:	learn: 0.6202508	test: 0.6199095	best: 0.6199095 (15)	total: 2m 55s	remaining: 15m 23s
    16:	learn: 0.6198238	test: 0.6194551	best: 0.6194551 (16)	total: 3m 7s	remaining: 15m 13s
    17:	learn: 0.6192924	test: 0.6189618	best: 0.6189618 (17)	total: 3m 16s	remaining: 14m 55s
    18:	learn: 0.6187464	test: 0.6184009	best: 0.6184009 (18)	total: 3m 26s	remaining: 14m 38s
    19:	learn: 0.6184773	test: 0.6181021	best: 0.6181021 (19)	total: 3m 37s	remaining: 14m 30s
    20:	learn: 0.6182807	test: 0.6179000	best: 0.6179000 (20)	total: 3m 47s	remaining: 14m 14s
    21:	learn: 0.6179597	test: 0.6175777	best: 0.6175777 (21)	total: 3m 56s	remaining: 13m 57s
    22:	learn: 0.6177107	test: 0.6173366	best: 0.6173366 (22)	total: 4m 5s	remaining: 13m 42s
    23:	learn: 0.6174131	test: 0.6170394	best: 0.6170394 (23)	total: 4m 14s	remaining: 13m 26s
    24:	learn: 0.6171068	test: 0.6167331	best: 0.6167331 (24)	total: 4m 23s	remaining: 13m 10s
    25:	learn: 0.6168559	test: 0.6164874	best: 0.6164874 (25)	total: 4m 32s	remaining: 12m 55s
    26:	learn: 0.6166080	test: 0.6162482	best: 0.6162482 (26)	total: 4m 41s	remaining: 12m 40s
    27:	learn: 0.6163412	test: 0.6159860	best: 0.6159860 (27)	total: 4m 50s	remaining: 12m 27s
    28:	learn: 0.6161286	test: 0.6157624	best: 0.6157624 (28)	total: 5m	remaining: 12m 16s
    29:	learn: 0.6157530	test: 0.6153989	best: 0.6153989 (29)	total: 5m 9s	remaining: 12m 2s
    30:	learn: 0.6155751	test: 0.6152261	best: 0.6152261 (30)	total: 5m 18s	remaining: 11m 48s
    31:	learn: 0.6151461	test: 0.6147762	best: 0.6147762 (31)	total: 5m 27s	remaining: 11m 35s
    32:	learn: 0.6149909	test: 0.6146237	best: 0.6146237 (32)	total: 5m 36s	remaining: 11m 23s
    33:	learn: 0.6147309	test: 0.6143658	best: 0.6143658 (33)	total: 5m 45s	remaining: 11m 9s
    34:	learn: 0.6145585	test: 0.6141879	best: 0.6141879 (34)	total: 5m 53s	remaining: 10m 57s
    35:	learn: 0.6143835	test: 0.6140231	best: 0.6140231 (35)	total: 6m 2s	remaining: 10m 44s
    36:	learn: 0.6141973	test: 0.6138201	best: 0.6138201 (36)	total: 6m 11s	remaining: 10m 32s
    37:	learn: 0.6139313	test: 0.6135377	best: 0.6135377 (37)	total: 6m 19s	remaining: 10m 19s
    38:	learn: 0.6137038	test: 0.6133267	best: 0.6133267 (38)	total: 6m 28s	remaining: 10m 7s
    39:	learn: 0.6135988	test: 0.6132291	best: 0.6132291 (39)	total: 6m 37s	remaining: 9m 56s
    40:	learn: 0.6134533	test: 0.6130749	best: 0.6130749 (40)	total: 6m 46s	remaining: 9m 44s
    41:	learn: 0.6131923	test: 0.6128131	best: 0.6128131 (41)	total: 6m 55s	remaining: 9m 33s
    42:	learn: 0.6130864	test: 0.6127251	best: 0.6127251 (42)	total: 7m 3s	remaining: 9m 21s
    43:	learn: 0.6128314	test: 0.6124790	best: 0.6124790 (43)	total: 7m 12s	remaining: 9m 10s
    44:	learn: 0.6126123	test: 0.6122577	best: 0.6122577 (44)	total: 7m 21s	remaining: 9m
    45:	learn: 0.6124265	test: 0.6120680	best: 0.6120680 (45)	total: 7m 30s	remaining: 8m 49s
    46:	learn: 0.6122817	test: 0.6119249	best: 0.6119249 (46)	total: 7m 39s	remaining: 8m 38s
    47:	learn: 0.6121628	test: 0.6118292	best: 0.6118292 (47)	total: 7m 48s	remaining: 8m 28s
    48:	learn: 0.6120618	test: 0.6117512	best: 0.6117512 (48)	total: 7m 58s	remaining: 8m 18s
    49:	learn: 0.6119226	test: 0.6116030	best: 0.6116030 (49)	total: 8m 8s	remaining: 8m 8s
    50:	learn: 0.6118006	test: 0.6114989	best: 0.6114989 (50)	total: 8m 17s	remaining: 7m 58s
    51:	learn: 0.6116835	test: 0.6113830	best: 0.6113830 (51)	total: 8m 26s	remaining: 7m 47s
    52:	learn: 0.6115544	test: 0.6112636	best: 0.6112636 (52)	total: 8m 35s	remaining: 7m 36s
    53:	learn: 0.6114265	test: 0.6111404	best: 0.6111404 (53)	total: 8m 43s	remaining: 7m 26s
    54:	learn: 0.6113012	test: 0.6110205	best: 0.6110205 (54)	total: 8m 52s	remaining: 7m 15s
    55:	learn: 0.6112065	test: 0.6109250	best: 0.6109250 (55)	total: 9m	remaining: 7m 4s
    56:	learn: 0.6111156	test: 0.6108328	best: 0.6108328 (56)	total: 9m 9s	remaining: 6m 54s
    57:	learn: 0.6110521	test: 0.6107601	best: 0.6107601 (57)	total: 9m 18s	remaining: 6m 44s
    58:	learn: 0.6109859	test: 0.6106919	best: 0.6106919 (58)	total: 9m 27s	remaining: 6m 34s
    59:	learn: 0.6108766	test: 0.6105805	best: 0.6105805 (59)	total: 9m 36s	remaining: 6m 24s
    60:	learn: 0.6107628	test: 0.6104765	best: 0.6104765 (60)	total: 9m 44s	remaining: 6m 14s
    61:	learn: 0.6106693	test: 0.6103896	best: 0.6103896 (61)	total: 9m 53s	remaining: 6m 3s
    62:	learn: 0.6105671	test: 0.6103023	best: 0.6103023 (62)	total: 10m 2s	remaining: 5m 53s
    63:	learn: 0.6104790	test: 0.6102005	best: 0.6102005 (63)	total: 10m 11s	remaining: 5m 43s
    64:	learn: 0.6104228	test: 0.6101550	best: 0.6101550 (64)	total: 10m 20s	remaining: 5m 33s
    65:	learn: 0.6103280	test: 0.6100315	best: 0.6100315 (65)	total: 10m 28s	remaining: 5m 23s
    66:	learn: 0.6102566	test: 0.6099706	best: 0.6099706 (66)	total: 10m 37s	remaining: 5m 14s
    67:	learn: 0.6101574	test: 0.6098614	best: 0.6098614 (67)	total: 10m 46s	remaining: 5m 4s
    68:	learn: 0.6101125	test: 0.6098310	best: 0.6098310 (68)	total: 10m 55s	remaining: 4m 54s
    69:	learn: 0.6099752	test: 0.6097115	best: 0.6097115 (69)	total: 11m 4s	remaining: 4m 44s
    70:	learn: 0.6098566	test: 0.6096095	best: 0.6096095 (70)	total: 11m 12s	remaining: 4m 34s
    71:	learn: 0.6097748	test: 0.6095382	best: 0.6095382 (71)	total: 11m 21s	remaining: 4m 24s
    72:	learn: 0.6096696	test: 0.6094216	best: 0.6094216 (72)	total: 11m 29s	remaining: 4m 15s
    73:	learn: 0.6095894	test: 0.6093563	best: 0.6093563 (73)	total: 11m 38s	remaining: 4m 5s
    74:	learn: 0.6095114	test: 0.6092762	best: 0.6092762 (74)	total: 11m 47s	remaining: 3m 55s
    75:	learn: 0.6094048	test: 0.6091737	best: 0.6091737 (75)	total: 11m 56s	remaining: 3m 46s
    76:	learn: 0.6092735	test: 0.6090479	best: 0.6090479 (76)	total: 12m 5s	remaining: 3m 36s
    77:	learn: 0.6091982	test: 0.6089699	best: 0.6089699 (77)	total: 12m 14s	remaining: 3m 27s
    78:	learn: 0.6090660	test: 0.6088366	best: 0.6088366 (78)	total: 12m 23s	remaining: 3m 17s
    79:	learn: 0.6089413	test: 0.6087154	best: 0.6087154 (79)	total: 12m 31s	remaining: 3m 7s
    80:	learn: 0.6088611	test: 0.6086329	best: 0.6086329 (80)	total: 12m 40s	remaining: 2m 58s
    81:	learn: 0.6088002	test: 0.6085739	best: 0.6085739 (81)	total: 12m 49s	remaining: 2m 48s
    82:	learn: 0.6087397	test: 0.6085315	best: 0.6085315 (82)	total: 12m 58s	remaining: 2m 39s
    83:	learn: 0.6086790	test: 0.6084761	best: 0.6084761 (83)	total: 13m 7s	remaining: 2m 29s
    84:	learn: 0.6086182	test: 0.6084103	best: 0.6084103 (84)	total: 13m 15s	remaining: 2m 20s
    85:	learn: 0.6085112	test: 0.6083172	best: 0.6083172 (85)	total: 13m 24s	remaining: 2m 10s
    86:	learn: 0.6084672	test: 0.6082801	best: 0.6082801 (86)	total: 13m 33s	remaining: 2m 1s
    87:	learn: 0.6083500	test: 0.6081723	best: 0.6081723 (87)	total: 13m 42s	remaining: 1m 52s
    88:	learn: 0.6081532	test: 0.6079737	best: 0.6079737 (88)	total: 13m 50s	remaining: 1m 42s
    89:	learn: 0.6080499	test: 0.6078734	best: 0.6078734 (89)	total: 13m 59s	remaining: 1m 33s
    90:	learn: 0.6079925	test: 0.6078206	best: 0.6078206 (90)	total: 14m 7s	remaining: 1m 23s
    91:	learn: 0.6079059	test: 0.6077291	best: 0.6077291 (91)	total: 14m 16s	remaining: 1m 14s
    92:	learn: 0.6078334	test: 0.6076693	best: 0.6076693 (92)	total: 14m 25s	remaining: 1m 5s
    93:	learn: 0.6077785	test: 0.6076222	best: 0.6076222 (93)	total: 14m 33s	remaining: 55.8s
    94:	learn: 0.6077253	test: 0.6075843	best: 0.6075843 (94)	total: 14m 42s	remaining: 46.5s
    95:	learn: 0.6076806	test: 0.6075396	best: 0.6075396 (95)	total: 14m 51s	remaining: 37.1s
    96:	learn: 0.6076225	test: 0.6074913	best: 0.6074913 (96)	total: 14m 59s	remaining: 27.8s
    97:	learn: 0.6075828	test: 0.6074595	best: 0.6074595 (97)	total: 15m 8s	remaining: 18.5s
    98:	learn: 0.6075180	test: 0.6074034	best: 0.6074034 (98)	total: 15m 17s	remaining: 9.27s
    99:	learn: 0.6073868	test: 0.6072815	best: 0.6072815 (99)	total: 15m 26s	remaining: 0us
    
    bestTest = 0.6072814833
    bestIteration = 99
    
    Wall time: 15min 59s
    




    <catboost.core.CatBoostClassifier at 0x16059ce9048>




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

    SmartScreen                                       		33.12626765699954
    AVProductStatesIdentifier                         		9.683154514497478
    AVProductsInstalled                               		8.033708984237073
    EngineVersion                                     		5.6583380023606855
    AppVersion                                        		4.622893722938757
    Census_ProcessorModelIdentifier                   		3.7843538593606847
    CountryIdentifier                                 		3.3487903638289795
    AvSigVersion                                      		3.013425562106083
    Wdft_IsGamer                                      		2.9370826006686612
    Census_OSInstallTypeName                          		2.9285523791723618
    Wdft_RegionIdentifier                             		1.6778611237575822
    Census_ActivationChannel                          		1.6208203649414876
    Census_OSBuildRevision                            		1.2779421228443004
    Census_OEMNameIdentifier                          		1.2008582261496472
    Census_OEMModelIdentifier                         		1.1748675015682768
    Census_PrimaryDiskTotalCapacity                   		1.1420550552579383
    OsBuild                                           		1.032566271208525
    DefaultBrowsersIdentifier                         		0.9031918029744861
    CityIdentifier                                    		0.7557648119364543
    RtpStateBitfield                                  		0.7495784726997435
    OsBuildLab                                        		0.6932582699761091
    Census_OSUILocaleIdentifier                       		0.690643369273425
    GeoNameIdentifier                                 		0.661299867491895
    Census_OSWUAutoUpdateOptionsName                  		0.6097691646464868
    Census_TotalPhysicalRAM                           		0.6048824684098099
    SkuEdition                                        		0.594356399387201
    Census_InternalPrimaryDiagonalDisplaySizeInInches 		0.5552051760169293
    LocaleEnglishNameIdentifier                       		0.5205601369732131
    Census_SystemVolumeTotalCapacity                  		0.48206197447662424
    Census_InternalBatteryNumberOfCharges             		0.4785074175264215
    IeVerIdentifier                                   		0.47800432630113676
    Census_GenuineStateName                           		0.4462569998256901
    Census_OSBranch                                   		0.3929802863184596
    Census_OSBuildNumber                              		0.3876832404112925
    Census_OSVersion                                  		0.30113145129038005
    Census_FirmwareVersionIdentifier                  		0.2763985987743505
    Census_InternalPrimaryDisplayResolutionHorizontal 		0.24976551604945896
    Census_HasOpticalDiskDrive                        		0.2311783327549624
    Processor                                         		0.22477480027016802
    Census_PrimaryDiskTypeName                        		0.19165475753152308
    Census_OSEdition                                  		0.1779653330629092
    Census_MDC2FormFactor                             		0.16879285740748515
    Census_IsAlwaysOnAlwaysConnectedCapable           		0.16466573422356615
    Census_IsSecureBootEnabled                        		0.1567699591510986
    IsProtected                                       		0.14492886746943753
    Census_FirmwareManufacturerIdentifier             		0.12479540597231119
    Platform                                          		0.11661943219086066
    Census_InternalPrimaryDisplayResolutionVertical   		0.1080282547453106
    Census_OSSkuName                                  		0.10331873032453877
    Census_InternalBatteryType                        		0.09922732537358049
    OrganizationIdentifier                            		0.09726065984659946
    Firewall                                          		0.0860422879071098
    Census_FlightRing                                 		0.08201925768115544
    OsSuite                                           		0.07951827001855408
    AVProductsEnabled                                 		0.07611161170858954
    OsVer                                             		0.0738183819518382
    Census_ChassisTypeName                            		0.06553640596607734
    Census_ProcessorClass                             		0.06446908842777757
    Census_OSInstallLanguageIdentifier                		0.05780098947098909
    Census_IsPenCapable                               		0.04974910202935887
    OsPlatformSubRelease                              		0.047973733878465846
    Census_ProcessorCoreCount                         		0.0382809653976343
    Census_PowerPlatformRoleName                      		0.03782363129345942
    Census_IsTouchEnabled                             		0.025704515376830713
    Census_ProcessorManufacturerIdentifier            		0.010333245910274076
    Census_OSArchitecture                             		0.0
    


```python
def plot_roc_curve(true,
                   pred,
                   name,
                   weights=None,
                   label='',
                   color='darkorange',
                   ax=None,
):
    if ax is None:
        ax = plt.gca()
    fpr, tpr, thr = roc_curve(true, pred, sample_weight=weights)
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
plt.figure(figsize=(10,10))
plot_roc_curve(true=val_pool.get_label(),
               pred=model.predict_proba(val_pool)[:,1],
               name='ROC-кривая на валидационном датасете')
```


![png](train_files/train_18_0.png)

