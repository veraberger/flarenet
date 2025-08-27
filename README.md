**flarenet** is a **convolutional neural network** used to **predict flares** in 20-second cadence data from NASA's Transiting Exoplanet Survey Satellite (**TESS**).

With a single line of code, you can generate flare predictions for any TESS 20-s target.

![Output predictions](https://github.com/veraberger/flarenet/blob/5f86206003fa66fc4a9390170b6ac45fdc9dfa39/figures/flare_predictions.png)

### Installation
With pip:
```bash
pip install flarenet20
```

For developers:
```bash
git clone https://github.com/veraberger/flarenet.git
cd flarenet
pip install .
```



### Usage
Tutorial notebook: [docs/example.ipynb](docs/example.ipynb)

Use our model trained on real TESS light curves with injected false-positives, or train your own with the ```Flarenet``` class.

The model outputs likelihoods only - choose your own confidence threshold for flares.


If you use this code in publications, please cite Berger, Schanche, et al. (2025).

MIT License
