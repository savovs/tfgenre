import os, json, errno
import pandas as pd
from pprint import pprint

file_names = []
file_paths = []
sound_categories = []

data_path = os.path.dirname(os.path.realpath(__file__)) + '/../data/'

max_files = 3000
count = 0
rows = []

print("Loading data...")
for root, sub_dirs, files in os.walk(data_path):
    for file in files:
        if (file.endswith('.json')):
            if count < max_files:
                file_names.append(file)

                path = os.path.join(root, file)
                file_paths.append(path)

                category = root.split('/')[-1]

                if (category not in sound_categories):
                    sound_categories.append(category)

                with open(path) as opened_file:
                    try:
                        dict_json = json.load(opened_file)

                    except Exception as err:
                        print(path, err)

                    # Remove unneeded keys
                    for key in ['metadata', 'sfx',]:
                        dict_json.pop(key, None)

                    # Encode label as number
                    dict_json['category'] = category # sound_categories.index(category)
                    
                    rows.append(dict_json)
            count += 1

data = pd.io.json.json_normalize(rows)


# Data saved as strings, will need to use literal_eval to parse .csv column
# Select whatever columns are needed here:
selected_columns = ['category', 'lowLevel.mfcc']

data = data[selected_columns]
data = data.set_index('category')
data.to_csv(data_path + 'data.csv')

try:
    os.makedirs(plot_path)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise
