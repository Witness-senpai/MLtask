import json
import os

from malware_src.model import load_model, apply_model_to_json

DEFAULT_MODEL_PATH = 'models/model.cb'


def change_location():
    script_location = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_location)


def main():
    change_location()
    print('Send json string to get prediction.')
    print('Send "stop" string to finish.')
    model = load_model(DEFAULT_MODEL_PATH)
    while True:
        input_string = input()
        if input_string == 'stop':
            break
        data = json.loads(input_string)
        prediction = apply_model_to_json(model, data)
        print(prediction)


if __name__ == '__main__':
    main()
