# This particular script runs on Python2.7 only.
#
# Extracts descriptive parameters from a folder with subfolders containing audio files
# where each subfolder is considered a category of sound.
#
# There needs to be an "audio" folder inside a "data" folder in the root directory.
#

import essentia, os, errno
from essentia.standard import *

file_names = []
file_paths = []
categories = []
file_extension = '.wav'
max_duration = 60 # seconds

# Get the audio directory path
audio_path = data_path = os.path.dirname(os.path.realpath(__file__)) + '/../data/audio'

# # Get the names and paths of all .file_extension files in the audio folder
for root, sub_dirs, files in os.walk(audio_path):
    for file in files:
        if (file.endswith(file_extension)):

            file_names.append(file)
            file_paths.append(os.path.join(root, file))

            category = root.split('/')[-1]

            if (category != 'data' and category not in categories):
                categories.append(category)

# Create result directories if necessary
result_root = os.path.dirname(os.path.realpath(__file__)) + '/../data/json'
result_directories = [os.path.join(result_root, category) for category in categories]

for path in result_directories:
    try:
        os.makedirs(path)
    except OSError as exception:
        if (exception.errno != errno.EEXIST):
            raise

result_paths = map(lambda path: path.replace(file_extension, '') + '.json', file_paths)
result_paths = map(lambda path: path.split('/audio')[1], result_paths)

print("\nAnalysing {} audio files shorter than {}s.\n".format(len(file_names), max_duration))

# Extract a bunch of audio features http://essentia.upf.edu/documentation/reference/std_Extractor.html
extractor = Extractor()
stats = ["min", "max", "median", "mean", "var", "cov", "kurt", "skew"]

numFiles = len(file_paths)
for index, path in enumerate(file_paths):
    # Load audio http://essentia.upf.edu/documentation/algorithms_overview.html#audio-input-output
    audio = MonoLoader(filename = path)()
    duration = Duration()(audio)

    # Ignore long files
    if (duration < max_duration):
        print("{0}, duration: {1:.2f}s, {2} files left".format(path.split('/audio')[1], duration, numFiles - index))
        current_extractor = extractor(audio)

        try:
            # # Statistical Aggregation http://essentia.upf.edu/documentation/reference/std_PoolAggregator.html
            # aggregated_pool = PoolAggregator(defaultStats = stats)(current_extractor)

            YamlOutput(filename = result_root + result_paths[index], format = 'json')(current_extractor)
        except Exception as e:
            print('\033[93m' + "Supressed error during statistical aggregation:", e)
            pass