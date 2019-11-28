import json
import os
import sys

from malware_src.model import load_model, apply_model_to_json
from malware_src.prepare_data import TARGET_COLUMN
from malware_src.database import DBLog

DEFAULT_MODEL_PATH = 'models/model.cb'
DEFAULT_DB_PATH = 'storage/log.db'
DEFAULT_CSVLOG_PATH = 'storage/log.csv'


def change_location():
    script_location = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_location)
    print(script_location)


def main():
    change_location()
    print('Send json string or json dictionary to get prediction.')
    print('Send "stop" string to finish.')
    model = load_model(DEFAULT_MODEL_PATH)
    # db = DBLog(DEFAULT_DB_PATH)
    # db.drop_table()
    with open(DEFAULT_CSVLOG_PATH, 'w') as log:
        log.write('prediction;target\n')
    while True:
        try:
            input_string = input()
        except EOFError:
            break
        if input_string == 'stop':
            break
        data = json.loads(input_string)

        # Таргет json-строки(для одиночных элементов) или json-словаря(> 1 элементов) явно
        # преобразуем в list, чтобы далее одинаково их обрабатывать.
        true_values = data.get(TARGET_COLUMN)
        if isinstance(true_values, dict):
            true_values = [true_values[key] for key in true_values.keys()]
        else:
            true_values = [true_values]

        scores = apply_model_to_json(model, data)
        with open(DEFAULT_CSVLOG_PATH, 'a') as log:
            for score, true_value in zip(scores, true_values):
                log.write(str(score) + ";" + str(true_value) + "\n")
        # db.insert(prediction, target)
    # db.close()


if __name__ == '__main__':
    main()
