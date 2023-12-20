**flarenet** is a **convolutional neural network** used to **predict flares** in 20-second cadence data from NASA's Transiting Exoplanet Survey Satellite (**TESS**).

In the final implementation of this project, a user will simply input a TESS observing target and some observing dates, and the model will download all relevant data and output predictions for each point in the light curve. 

To make predictions, the model uses flux light curves with additional information from the centroid and position correction light curves, cosmic rays, and intrinsic properties of the stars themselves.
![Network input](https://github.com/veraberger/flarenet/blob/5f86206003fa66fc4a9390170b6ac45fdc9dfa39/figures/network_inputs.png)

Here's an example output of a light curve, color coded by the likelihood that each point is part of a flare (brighter color = higher likelihood)
![Output predictions](https://github.com/veraberger/flarenet/blob/5f86206003fa66fc4a9390170b6ac45fdc9dfa39/figures/flare_predictions.png)



**notes:**

-- this project was developed as part of an OSTEM internship at NASA Goddard with the TESS GI team!

-- name suggestions welcome. `flarenet` is not final.

-- does this project sound familiar? 
[stella](https://github.com/afeinstein20/stella) is a network developed for flare detection with 2-minute cadence observations from TESS. However, the shorter-cadence data unveils more detailed light curve morphology. Below is an example of light curves for the same flare; note the high flux and double peak resolved by the shorter-cadence observation! 
![20-s vs 2-min cadence](https://github.com/veraberger/flarenet/blob/5f86206003fa66fc4a9390170b6ac45fdc9dfa39/figures/tess_flares_20s_2min.png)

