# Импорт библиотек


```python
import gc

import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
sns.set(style="ticks", color_codes=True)
```

# Анализ данных

Источник данных: https://www.kaggle.com/c/microsoft-malware-prediction/data


```python
data_path = '../data/train.csv'
data_submission_path = '../data/sample_submission.csv'
```


```python
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
data = pd.read_csv(data_path, dtype=dtypes)
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MachineIdentifier</th>
      <th>ProductName</th>
      <th>EngineVersion</th>
      <th>AppVersion</th>
      <th>AvSigVersion</th>
      <th>IsBeta</th>
      <th>RtpStateBitfield</th>
      <th>IsSxsPassiveMode</th>
      <th>DefaultBrowsersIdentifier</th>
      <th>AVProductStatesIdentifier</th>
      <th>...</th>
      <th>Census_FirmwareVersionIdentifier</th>
      <th>Census_IsSecureBootEnabled</th>
      <th>Census_IsWIMBootEnabled</th>
      <th>Census_IsVirtualDevice</th>
      <th>Census_IsTouchEnabled</th>
      <th>Census_IsPenCapable</th>
      <th>Census_IsAlwaysOnAlwaysConnectedCapable</th>
      <th>Wdft_IsGamer</th>
      <th>Wdft_RegionIdentifier</th>
      <th>HasDetections</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0000028988387b115f69f31a3bf04f09</td>
      <td>win8defender</td>
      <td>1.1.15100.1</td>
      <td>4.18.1807.18075</td>
      <td>1.273.1735.0</td>
      <td>0</td>
      <td>7.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>53447.0</td>
      <td>...</td>
      <td>36144.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>000007535c3f730efa9ea0b7ef1bd645</td>
      <td>win8defender</td>
      <td>1.1.14600.4</td>
      <td>4.13.17134.1</td>
      <td>1.263.48.0</td>
      <td>0</td>
      <td>7.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>53447.0</td>
      <td>...</td>
      <td>57858.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>000007905a28d863f6d0d597892cd692</td>
      <td>win8defender</td>
      <td>1.1.15100.1</td>
      <td>4.18.1807.18075</td>
      <td>1.273.1341.0</td>
      <td>0</td>
      <td>7.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>53447.0</td>
      <td>...</td>
      <td>52682.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>00000b11598a75ea8ba1beea8459149f</td>
      <td>win8defender</td>
      <td>1.1.15100.1</td>
      <td>4.18.1807.18075</td>
      <td>1.273.1527.0</td>
      <td>0</td>
      <td>7.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>53447.0</td>
      <td>...</td>
      <td>20050.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>000014a5f00daa18e76b81417eeb99fc</td>
      <td>win8defender</td>
      <td>1.1.15100.1</td>
      <td>4.18.1807.18075</td>
      <td>1.273.1379.0</td>
      <td>0</td>
      <td>7.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>53447.0</td>
      <td>...</td>
      <td>19844.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 83 columns</p>
</div>




```python
data['HasDetections'].value_counts()
```




    0    4462591
    1    4458892
    Name: HasDetections, dtype: int64




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8921483 entries, 0 to 8921482
    Data columns (total 83 columns):
    MachineIdentifier                                    category
    ProductName                                          category
    EngineVersion                                        category
    AppVersion                                           category
    AvSigVersion                                         category
    IsBeta                                               int8
    RtpStateBitfield                                     float16
    IsSxsPassiveMode                                     int8
    DefaultBrowsersIdentifier                            float32
    AVProductStatesIdentifier                            float32
    AVProductsInstalled                                  float16
    AVProductsEnabled                                    float16
    HasTpm                                               int8
    CountryIdentifier                                    int16
    CityIdentifier                                       float32
    OrganizationIdentifier                               float16
    GeoNameIdentifier                                    float16
    LocaleEnglishNameIdentifier                          int16
    Platform                                             category
    Processor                                            category
    OsVer                                                category
    OsBuild                                              int16
    OsSuite                                              int16
    OsPlatformSubRelease                                 category
    OsBuildLab                                           category
    SkuEdition                                           category
    IsProtected                                          float16
    AutoSampleOptIn                                      int8
    PuaMode                                              category
    SMode                                                float16
    IeVerIdentifier                                      float16
    SmartScreen                                          category
    Firewall                                             float16
    UacLuaenable                                         float64
    Census_MDC2FormFactor                                category
    Census_DeviceFamily                                  category
    Census_OEMNameIdentifier                             float32
    Census_OEMModelIdentifier                            float32
    Census_ProcessorCoreCount                            float16
    Census_ProcessorManufacturerIdentifier               float16
    Census_ProcessorModelIdentifier                      float32
    Census_ProcessorClass                                category
    Census_PrimaryDiskTotalCapacity                      float64
    Census_PrimaryDiskTypeName                           category
    Census_SystemVolumeTotalCapacity                     float64
    Census_HasOpticalDiskDrive                           int8
    Census_TotalPhysicalRAM                              float32
    Census_ChassisTypeName                               category
    Census_InternalPrimaryDiagonalDisplaySizeInInches    float32
    Census_InternalPrimaryDisplayResolutionHorizontal    float32
    Census_InternalPrimaryDisplayResolutionVertical      float32
    Census_PowerPlatformRoleName                         category
    Census_InternalBatteryType                           category
    Census_InternalBatteryNumberOfCharges                float64
    Census_OSVersion                                     category
    Census_OSArchitecture                                category
    Census_OSBranch                                      category
    Census_OSBuildNumber                                 int16
    Census_OSBuildRevision                               int32
    Census_OSEdition                                     category
    Census_OSSkuName                                     category
    Census_OSInstallTypeName                             category
    Census_OSInstallLanguageIdentifier                   float16
    Census_OSUILocaleIdentifier                          int16
    Census_OSWUAutoUpdateOptionsName                     category
    Census_IsPortableOperatingSystem                     int8
    Census_GenuineStateName                              category
    Census_ActivationChannel                             category
    Census_IsFlightingInternal                           float16
    Census_IsFlightsDisabled                             float16
    Census_FlightRing                                    category
    Census_ThresholdOptIn                                float16
    Census_FirmwareManufacturerIdentifier                float16
    Census_FirmwareVersionIdentifier                     float32
    Census_IsSecureBootEnabled                           int8
    Census_IsWIMBootEnabled                              float16
    Census_IsVirtualDevice                               float16
    Census_IsTouchEnabled                                int8
    Census_IsPenCapable                                  int8
    Census_IsAlwaysOnAlwaysConnectedCapable              float16
    Wdft_IsGamer                                         float16
    Wdft_RegionIdentifier                                float16
    HasDetections                                        int8
    dtypes: category(30), float16(21), float32(11), float64(4), int16(6), int32(1), int8(10)
    memory usage: 1.9 GB
    


```python
cols = data.columns.to_list()
```


```python
print(*cols, sep="\n")
```

    MachineIdentifier
    ProductName
    EngineVersion
    AppVersion
    AvSigVersion
    IsBeta
    RtpStateBitfield
    IsSxsPassiveMode
    DefaultBrowsersIdentifier
    AVProductStatesIdentifier
    AVProductsInstalled
    AVProductsEnabled
    HasTpm
    CountryIdentifier
    CityIdentifier
    OrganizationIdentifier
    GeoNameIdentifier
    LocaleEnglishNameIdentifier
    Platform
    Processor
    OsVer
    OsBuild
    OsSuite
    OsPlatformSubRelease
    OsBuildLab
    SkuEdition
    IsProtected
    AutoSampleOptIn
    PuaMode
    SMode
    IeVerIdentifier
    SmartScreen
    Firewall
    UacLuaenable
    Census_MDC2FormFactor
    Census_DeviceFamily
    Census_OEMNameIdentifier
    Census_OEMModelIdentifier
    Census_ProcessorCoreCount
    Census_ProcessorManufacturerIdentifier
    Census_ProcessorModelIdentifier
    Census_ProcessorClass
    Census_PrimaryDiskTotalCapacity
    Census_PrimaryDiskTypeName
    Census_SystemVolumeTotalCapacity
    Census_HasOpticalDiskDrive
    Census_TotalPhysicalRAM
    Census_ChassisTypeName
    Census_InternalPrimaryDiagonalDisplaySizeInInches
    Census_InternalPrimaryDisplayResolutionHorizontal
    Census_InternalPrimaryDisplayResolutionVertical
    Census_PowerPlatformRoleName
    Census_InternalBatteryType
    Census_InternalBatteryNumberOfCharges
    Census_OSVersion
    Census_OSArchitecture
    Census_OSBranch
    Census_OSBuildNumber
    Census_OSBuildRevision
    Census_OSEdition
    Census_OSSkuName
    Census_OSInstallTypeName
    Census_OSInstallLanguageIdentifier
    Census_OSUILocaleIdentifier
    Census_OSWUAutoUpdateOptionsName
    Census_IsPortableOperatingSystem
    Census_GenuineStateName
    Census_ActivationChannel
    Census_IsFlightingInternal
    Census_IsFlightsDisabled
    Census_FlightRing
    Census_ThresholdOptIn
    Census_FirmwareManufacturerIdentifier
    Census_FirmwareVersionIdentifier
    Census_IsSecureBootEnabled
    Census_IsWIMBootEnabled
    Census_IsVirtualDevice
    Census_IsTouchEnabled
    Census_IsPenCapable
    Census_IsAlwaysOnAlwaysConnectedCapable
    Wdft_IsGamer
    Wdft_RegionIdentifier
    HasDetections
    


```python
data.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>IsBeta</td>
      <td>8921483.0</td>
      <td>7.509962e-06</td>
      <td>2.740421e-03</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>RtpStateBitfield</td>
      <td>8889165.0</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>7.000000e+00</td>
      <td>3.500000e+01</td>
    </tr>
    <tr>
      <td>IsSxsPassiveMode</td>
      <td>8921483.0</td>
      <td>1.733378e-02</td>
      <td>1.305118e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>DefaultBrowsersIdentifier</td>
      <td>433438.0</td>
      <td>1.658356e+03</td>
      <td>9.989604e+02</td>
      <td>1.0</td>
      <td>788.0</td>
      <td>1632.0</td>
      <td>2.373000e+03</td>
      <td>3.213000e+03</td>
    </tr>
    <tr>
      <td>AVProductStatesIdentifier</td>
      <td>8885262.0</td>
      <td>4.784002e+04</td>
      <td>1.403237e+04</td>
      <td>3.0</td>
      <td>49480.0</td>
      <td>53447.0</td>
      <td>5.344700e+04</td>
      <td>7.050700e+04</td>
    </tr>
    <tr>
      <td>AVProductsInstalled</td>
      <td>8885262.0</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.000000e+00</td>
      <td>7.000000e+00</td>
    </tr>
    <tr>
      <td>AVProductsEnabled</td>
      <td>8885262.0</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000e+00</td>
      <td>5.000000e+00</td>
    </tr>
    <tr>
      <td>HasTpm</td>
      <td>8921483.0</td>
      <td>9.879711e-01</td>
      <td>1.090149e-01</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>CountryIdentifier</td>
      <td>8921483.0</td>
      <td>1.080490e+02</td>
      <td>6.304706e+01</td>
      <td>1.0</td>
      <td>51.0</td>
      <td>97.0</td>
      <td>1.620000e+02</td>
      <td>2.220000e+02</td>
    </tr>
    <tr>
      <td>CityIdentifier</td>
      <td>8596074.0</td>
      <td>8.126650e+04</td>
      <td>4.892339e+04</td>
      <td>5.0</td>
      <td>36825.0</td>
      <td>82373.0</td>
      <td>1.237000e+05</td>
      <td>1.679620e+05</td>
    </tr>
    <tr>
      <td>OrganizationIdentifier</td>
      <td>6169965.0</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>18.0</td>
      <td>27.0</td>
      <td>2.700000e+01</td>
      <td>5.200000e+01</td>
    </tr>
    <tr>
      <td>GeoNameIdentifier</td>
      <td>8921270.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>89.0</td>
      <td>181.0</td>
      <td>2.670000e+02</td>
      <td>2.960000e+02</td>
    </tr>
    <tr>
      <td>LocaleEnglishNameIdentifier</td>
      <td>8921483.0</td>
      <td>1.228161e+02</td>
      <td>6.932125e+01</td>
      <td>1.0</td>
      <td>74.0</td>
      <td>88.0</td>
      <td>1.820000e+02</td>
      <td>2.830000e+02</td>
    </tr>
    <tr>
      <td>OsBuild</td>
      <td>8921483.0</td>
      <td>1.571997e+04</td>
      <td>2.190685e+03</td>
      <td>7600.0</td>
      <td>15063.0</td>
      <td>16299.0</td>
      <td>1.713400e+04</td>
      <td>1.824400e+04</td>
    </tr>
    <tr>
      <td>OsSuite</td>
      <td>8921483.0</td>
      <td>5.751534e+02</td>
      <td>2.480847e+02</td>
      <td>16.0</td>
      <td>256.0</td>
      <td>768.0</td>
      <td>7.680000e+02</td>
      <td>7.840000e+02</td>
    </tr>
    <tr>
      <td>IsProtected</td>
      <td>8885439.0</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>AutoSampleOptIn</td>
      <td>8921483.0</td>
      <td>2.891896e-05</td>
      <td>5.377558e-03</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>SMode</td>
      <td>8383724.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>IeVerIdentifier</td>
      <td>8862589.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>111.0</td>
      <td>117.0</td>
      <td>1.370000e+02</td>
      <td>4.290000e+02</td>
    </tr>
    <tr>
      <td>Firewall</td>
      <td>8830133.0</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>UacLuaenable</td>
      <td>8910645.0</td>
      <td>1.302773e+01</td>
      <td>9.867770e+03</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000e+00</td>
      <td>1.677722e+07</td>
    </tr>
    <tr>
      <td>Census_OEMNameIdentifier</td>
      <td>8826005.0</td>
      <td>2.220166e+03</td>
      <td>1.315713e+03</td>
      <td>1.0</td>
      <td>1443.0</td>
      <td>2102.0</td>
      <td>2.668000e+03</td>
      <td>6.145000e+03</td>
    </tr>
    <tr>
      <td>Census_OEMModelIdentifier</td>
      <td>8819250.0</td>
      <td>2.391425e+05</td>
      <td>7.194786e+04</td>
      <td>1.0</td>
      <td>189692.0</td>
      <td>247458.0</td>
      <td>3.044180e+05</td>
      <td>3.454980e+05</td>
    </tr>
    <tr>
      <td>Census_ProcessorCoreCount</td>
      <td>8880177.0</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.000000e+00</td>
      <td>1.920000e+02</td>
    </tr>
    <tr>
      <td>Census_ProcessorManufacturerIdentifier</td>
      <td>8880170.0</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.000000e+00</td>
      <td>1.000000e+01</td>
    </tr>
    <tr>
      <td>Census_ProcessorModelIdentifier</td>
      <td>8880140.0</td>
      <td>2.371274e+03</td>
      <td>8.406009e+02</td>
      <td>2.0</td>
      <td>1998.0</td>
      <td>2500.0</td>
      <td>2.874000e+03</td>
      <td>4.479000e+03</td>
    </tr>
    <tr>
      <td>Census_PrimaryDiskTotalCapacity</td>
      <td>8868467.0</td>
      <td>3.089053e+06</td>
      <td>4.451634e+09</td>
      <td>0.0</td>
      <td>239372.0</td>
      <td>476940.0</td>
      <td>9.538690e+05</td>
      <td>8.160437e+12</td>
    </tr>
    <tr>
      <td>Census_SystemVolumeTotalCapacity</td>
      <td>8868481.0</td>
      <td>3.773683e+05</td>
      <td>3.258791e+05</td>
      <td>0.0</td>
      <td>120775.0</td>
      <td>249500.0</td>
      <td>4.759730e+05</td>
      <td>4.768710e+07</td>
    </tr>
    <tr>
      <td>Census_HasOpticalDiskDrive</td>
      <td>8921483.0</td>
      <td>7.718728e-02</td>
      <td>2.668884e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>Census_TotalPhysicalRAM</td>
      <td>8840950.0</td>
      <td>6.115257e+03</td>
      <td>5.115821e+03</td>
      <td>255.0</td>
      <td>4096.0</td>
      <td>4096.0</td>
      <td>8.192000e+03</td>
      <td>1.572864e+06</td>
    </tr>
    <tr>
      <td>Census_InternalPrimaryDiagonalDisplaySizeInInches</td>
      <td>8874349.0</td>
      <td>1.667620e+01</td>
      <td>5.892932e+00</td>
      <td>0.7</td>
      <td>13.9</td>
      <td>15.5</td>
      <td>1.720000e+01</td>
      <td>1.823000e+02</td>
    </tr>
    <tr>
      <td>Census_InternalPrimaryDisplayResolutionHorizontal</td>
      <td>8874497.0</td>
      <td>1.547716e+03</td>
      <td>3.683716e+02</td>
      <td>-1.0</td>
      <td>1366.0</td>
      <td>1366.0</td>
      <td>1.920000e+03</td>
      <td>1.228800e+04</td>
    </tr>
    <tr>
      <td>Census_InternalPrimaryDisplayResolutionVertical</td>
      <td>8874497.0</td>
      <td>8.975703e+02</td>
      <td>2.146239e+02</td>
      <td>-1.0</td>
      <td>768.0</td>
      <td>768.0</td>
      <td>1.080000e+03</td>
      <td>8.640000e+03</td>
    </tr>
    <tr>
      <td>Census_InternalBatteryNumberOfCharges</td>
      <td>8652728.0</td>
      <td>1.123782e+09</td>
      <td>1.887782e+09</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.294967e+09</td>
      <td>4.294967e+09</td>
    </tr>
    <tr>
      <td>Census_OSBuildNumber</td>
      <td>8921483.0</td>
      <td>1.583483e+04</td>
      <td>1.961743e+03</td>
      <td>7600.0</td>
      <td>15063.0</td>
      <td>16299.0</td>
      <td>1.713400e+04</td>
      <td>1.824400e+04</td>
    </tr>
    <tr>
      <td>Census_OSBuildRevision</td>
      <td>8921483.0</td>
      <td>9.730490e+02</td>
      <td>2.931971e+03</td>
      <td>0.0</td>
      <td>167.0</td>
      <td>285.0</td>
      <td>5.470000e+02</td>
      <td>4.173600e+04</td>
    </tr>
    <tr>
      <td>Census_OSInstallLanguageIdentifier</td>
      <td>8861399.0</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>2.000000e+01</td>
      <td>3.900000e+01</td>
    </tr>
    <tr>
      <td>Census_OSUILocaleIdentifier</td>
      <td>8921483.0</td>
      <td>6.046534e+01</td>
      <td>4.499992e+01</td>
      <td>1.0</td>
      <td>31.0</td>
      <td>34.0</td>
      <td>9.000000e+01</td>
      <td>1.620000e+02</td>
    </tr>
    <tr>
      <td>Census_IsPortableOperatingSystem</td>
      <td>8921483.0</td>
      <td>5.452008e-04</td>
      <td>2.334317e-02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>Census_IsFlightingInternal</td>
      <td>1512724.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>Census_IsFlightsDisabled</td>
      <td>8760960.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>Census_ThresholdOptIn</td>
      <td>3254158.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>Census_FirmwareManufacturerIdentifier</td>
      <td>8738226.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>142.0</td>
      <td>500.0</td>
      <td>5.560000e+02</td>
      <td>1.092000e+03</td>
    </tr>
    <tr>
      <td>Census_FirmwareVersionIdentifier</td>
      <td>8761350.0</td>
      <td>3.302793e+04</td>
      <td>2.120691e+04</td>
      <td>3.0</td>
      <td>13156.0</td>
      <td>33070.0</td>
      <td>5.243600e+04</td>
      <td>7.210500e+04</td>
    </tr>
    <tr>
      <td>Census_IsSecureBootEnabled</td>
      <td>8921483.0</td>
      <td>4.860229e-01</td>
      <td>4.998046e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>Census_IsWIMBootEnabled</td>
      <td>3261780.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>Census_IsVirtualDevice</td>
      <td>8905530.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>Census_IsTouchEnabled</td>
      <td>8921483.0</td>
      <td>1.255431e-01</td>
      <td>3.313338e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>Census_IsPenCapable</td>
      <td>8921483.0</td>
      <td>3.807091e-02</td>
      <td>1.913675e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>Census_IsAlwaysOnAlwaysConnectedCapable</td>
      <td>8850140.0</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>Wdft_IsGamer</td>
      <td>8618032.0</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <td>Wdft_RegionIdentifier</td>
      <td>8618032.0</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>10.0</td>
      <td>1.100000e+01</td>
      <td>1.500000e+01</td>
    </tr>
    <tr>
      <td>HasDetections</td>
      <td>8921483.0</td>
      <td>4.997927e-01</td>
      <td>5.000000e-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
  </tbody>
</table>
</div>




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

# Analisis each column and grouping by numeric and catecorical features
for col in cols[1:]:
    col_stat = data[col].value_counts()
    unique_values = len(col_stat)
    part_most_popular_val = col_stat.iloc[0] / col_stat.sum()
    stat_cols.append((col, unique_values, part_most_popular_val))
    
    if (col not in numerical_cols and part_most_popular_val <= 0.98):
        categorical_cols.append(col)

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

    name                                               unique values  part of most popular  
    ProductName                                        6              0.9893556934424468    
    EngineVersion                                      70             0.43098966842171865   
    AppVersion                                         110            0.5760504167300436    
    AvSigVersion                                       8531           0.011468608974539322  
    IsBeta                                             2              0.999992490037811     
    RtpStateBitfield                                   7              0.9732620555473995    
    IsSxsPassiveMode                                   2              0.9826662226448226    
    DefaultBrowsersIdentifier                          2017           0.10625741167133478   
    AVProductStatesIdentifier                          28970          0.6555310355507806    
    AVProductsInstalled                                8              0.6987855844881108    
    AVProductsEnabled                                  6              0.9739837722286636    
    HasTpm                                             2              0.9879710581749693    
    CountryIdentifier                                  222            0.04451860750056913   
    CityIdentifier                                     107366         0.01102968634285838   
    OrganizationIdentifier                             49             0.6801427560772225    
    GeoNameIdentifier                                  292            0.17171647086121147   
    LocaleEnglishNameIdentifier                        276            0.23477991271182155   
    Platform                                           4              0.9660630413127503    
    Processor                                          3              0.9085300056055703    
    OsVer                                              58             0.9676132320153499    
    OsBuild                                            76             0.43888678597493264   
    OsSuite                                            14             0.6232888635219055    
    OsPlatformSubRelease                               9              0.43888734641987215   
    OsBuildLab                                         663            0.41004478862321      
    SkuEdition                                         8              0.6180969016025699    
    IsProtected                                        2              0.9456237333912259    
    AutoSampleOptIn                                    2              0.9999710810411229    
    PuaMode                                            2              0.9991338241663058    
    SMode                                              2              0.9995370792263677    
    IeVerIdentifier                                    303            0.4384544967616122    
    SmartScreen                                        21             0.7513628754740209    
    Firewall                                           2              0.978582542301458     
    UacLuaenable                                       11             0.9939254677972246    
    Census_MDC2FormFactor                              13             0.6415210341150681    
    Census_DeviceFamily                                3              0.998382555904663     
    Census_OEMNameIdentifier                           3832           0.14585024594932816   
    Census_OEMModelIdentifier                          175365         0.03455872097967514   
    Census_ProcessorCoreCount                          45             0.6114960321173778    
    Census_ProcessorManufacturerIdentifier             7              0.8827891808377543    
    Census_ProcessorModelIdentifier                    3428           0.03257640082250955   
    Census_ProcessorClass                              3              0.5709371843520515    
    Census_PrimaryDiskTotalCapacity                    5735           0.32040825094122805   
    Census_PrimaryDiskTypeName                         4              0.6518171855431565    
    Census_SystemVolumeTotalCapacity                   536848         0.005863236331001893  
    Census_HasOpticalDiskDrive                         2              0.9228127207102227    
    Census_TotalPhysicalRAM                            3446           0.46313031970546154   
    Census_ChassisTypeName                             52             0.5883751118165738    
    Census_InternalPrimaryDiagonalDisplaySizeInInches  785            0.34339769598874237   
    Census_InternalPrimaryDisplayResolutionHorizontal  2180           0.5087684406226065    
    Census_InternalPrimaryDisplayResolutionVertical    1560           0.5604397635156111    
    Census_PowerPlatformRoleName                       10             0.6930401724925651    
    Census_InternalBatteryType                         78             0.7852162595129641    
    Census_InternalBatteryNumberOfCharges              41088          0.584024367806315     
    Census_OSVersion                                   469            0.15845201969224174   
    Census_OSArchitecture                              3              0.9085804456501234    
    Census_OSBranch                                    32             0.449382462534536     
    Census_OSBuildNumber                               165            0.44935141388488886   
    Census_OSBuildRevision                             285            0.15845269222616912   
    Census_OSEdition                                   33             0.3889477791976962    
    Census_OSSkuName                                   30             0.38893410434117287   
    Census_OSInstallTypeName                           9              0.29233222772491974   
    Census_OSInstallLanguageIdentifier                 39             0.3587765317869108    
    Census_OSUILocaleIdentifier                        147            0.3554144529558595    
    Census_OSWUAutoUpdateOptionsName                   6              0.4432555663671612    
    Census_IsPortableOperatingSystem                   2              0.9994547991628746    
    Census_GenuineStateName                            5              0.8829918747813564    
    Census_ActivationChannel                           6              0.5299106661975369    
    Census_IsFlightingInternal                         2              0.9999861177584278    
    Census_IsFlightsDisabled                           2              0.9999899554386734    
    Census_FlightRing                                  10             0.9365796022925785    
    Census_ThresholdOptIn                              2              0.9997492438904318    
    Census_FirmwareManufacturerIdentifier              712            0.3088816883426911    
    Census_FirmwareVersionIdentifier                   50494          0.010227989978713325  
    Census_IsSecureBootEnabled                         2              0.5139771044791545    
    Census_IsWIMBootEnabled                            2              0.9999996934189308    
    Census_IsVirtualDevice                             2              0.9929605537233607    
    Census_IsTouchEnabled                              2              0.8744568587980271    
    Census_IsPenCapable                                2              0.9619290873501637    
    Census_IsAlwaysOnAlwaysConnectedCapable            2              0.9425807953320512    
    Wdft_IsGamer                                       2              0.7164214521366363    
    Wdft_RegionIdentifier                              15             0.20887657414128888   
    HasDetections                                      2              0.5002073085831134    
    


```python
print(f'Categorical_cols ({len(categorical_cols)} pieces):')
print(*categorical_cols, sep='\n')
```

    Categorical_cols (59 pieces):
    EngineVersion
    AppVersion
    AvSigVersion
    RtpStateBitfield
    DefaultBrowsersIdentifier
    AVProductStatesIdentifier
    AVProductsInstalled
    AVProductsEnabled
    CountryIdentifier
    CityIdentifier
    OrganizationIdentifier
    GeoNameIdentifier
    LocaleEnglishNameIdentifier
    Platform
    Processor
    OsVer
    OsBuild
    OsSuite
    OsPlatformSubRelease
    OsBuildLab
    SkuEdition
    IsProtected
    IeVerIdentifier
    SmartScreen
    Firewall
    Census_MDC2FormFactor
    Census_OEMNameIdentifier
    Census_OEMModelIdentifier
    Census_ProcessorManufacturerIdentifier
    Census_ProcessorModelIdentifier
    Census_ProcessorClass
    Census_PrimaryDiskTypeName
    Census_HasOpticalDiskDrive
    Census_ChassisTypeName
    Census_PowerPlatformRoleName
    Census_InternalBatteryType
    Census_OSVersion
    Census_OSArchitecture
    Census_OSBranch
    Census_OSBuildNumber
    Census_OSBuildRevision
    Census_OSEdition
    Census_OSSkuName
    Census_OSInstallTypeName
    Census_OSInstallLanguageIdentifier
    Census_OSUILocaleIdentifier
    Census_OSWUAutoUpdateOptionsName
    Census_GenuineStateName
    Census_ActivationChannel
    Census_FlightRing
    Census_FirmwareManufacturerIdentifier
    Census_FirmwareVersionIdentifier
    Census_IsSecureBootEnabled
    Census_IsTouchEnabled
    Census_IsPenCapable
    Census_IsAlwaysOnAlwaysConnectedCapable
    Wdft_IsGamer
    Wdft_RegionIdentifier
    HasDetections
    

# Сохраняем данные только с необходимыми столбцами


```python
filtered_cols = numerical_cols + categorical_cols

for col in cols:
    if col not in filtered_cols:
        data.drop(col, axis=1, inplace=True)
```


```python
data.to_csv('../data/filtered_train_data.csv', index=False)
```

# Небольшая визуализация


```python
_ = data[numerical_cols].hist(figsize=(35,35))
```


![png](data_exploring_files/data_exploring_19_0.png)



```python
_2 = data[categorical_cols].hist(figsize=(35,35))
```


![png](data_exploring_files/data_exploring_20_0.png)

