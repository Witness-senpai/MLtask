# Пропуск отложенных данных через обученную модель

### Импорты


```python
import os
import json
import platform

import pandas as pd
```

### Подготовка json данных


```python
data_path = 'data/hold_data.csv'
json_input_path = 'data/dict_input.json'
strings_input_path = 'data/strings_input.txt'
script_path = 'detector.py'
```


```python
data = pd.read_csv(data_path)

print(data.shape)

# Данные в большой json-словарь
data.to_json(json_input_path)
```


```python
# Конвертация всех данных в лист json-строк
json_strings = []

for row in data.iterrows():
    json_string = json.dumps(row[1].to_dict())
    json_strings.append(json_string)
    
print(f'Rows of data: {len(json_strings)}\n')
print(json_strings[0])
```


```python
input_string = '\n'.join(json_strings) + '\nstop\n'

print(len(input_string))

# Запись полученных данных в файл
with open(strings_input_path, 'w') as f:
    f.write(input_string)
```

### Пропуск данных черех скрипт


```python
# Команда в зависимости от платформы
if platform.system() == 'Windows':
    python_path = 'python'
    execute_comand = f'type {json_input_path} | {python_path} {script_path}'
else: # Linux
    python_path = 'python3'
    execute_comand = f'cat {json_input_path} | {python_path} {script_path}'
print(execute_comand)
```


```python
%%time
ans = os.system(execute_comand)
```
