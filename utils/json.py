import json

def load(filepath):
    with open(filepath, 'r') as file:
        content = json.load(file)

    return content