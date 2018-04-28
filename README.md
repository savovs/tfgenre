# tfgenre
### To evaluate accuracy of all models:
```sh
sh evaluate.sh
```

### To train all models
```sh
sh train.sh
```

### Run TensorBoard for visualisations (replace with absolute path to logs folder):
```sh
tensorboard --logdir=/path/to/tfgenre/logs
```

#### Optional audio data extraction guide with [Essentia](http://essentia.upf.edu/documentation/) (macOS only)
Data set used for this project can be found [here](http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html)

1. Install Essentia with [Homebrew](https://brew.sh/): `brew tap MTG/essentia && brew install essentia`
2. Make a `data/audio` folder in the project root containing subfolders of audio files.
Subfolder names used as categories, i.e. `data/audio/car_horn/11251.wav`

3. Run `src/extract_features.py` with Python2.7 to load audio from `data/audio` and produce JSON files with audio features
4. Run `write_csv.py` to distill all JSON to one .csv
