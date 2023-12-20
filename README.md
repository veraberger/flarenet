**flarenet** is a **convolutional neural network** used to **predict flares** in 20-second cadence data from NASA's Transiting Exoplanet Survey Satellite (**TESS**).

In the final implementation of this project, a user will simply inputs a TESS observing target and some observing dates, and the model will download all relevant data and output predictions for each point in the light curve. 

![Network input](https://drive.google.com/file/d/1fNSjTGGgvoQ9IyuBuSEEZHzJ4c-dYEuy/view?usp=sharing)
![Output predictions](https://drive.google.com/file/d/14ZhK4zzso1tntHw4ox2ZHBIhlcSkTsRe/view?usp=sharing)



**notes:**
this project was developed as part of an OSTEM internship at NASA Goddard with the TESS GI team!

name suggestions welcome. `flarenet` is not final.

does this project sound familiar? 
[stella](https://github.com/afeinstein20/stella) is a network developed for flare detection with 2-minute cadence observations from TESS. However, the shorter-cadence data unveils more detailed light curve morphology. Below is an example of light curves for the same flare from TESS; 2-minute cadence on the left and 20-second on the right. 
![20-s vs 2-min cadence](https://drive.google.com/file/d/1WOvRVfRf-PzNH6ocOy-ZZwRi1conS3_X/view?usp=sharing)

