from . import PACKAGEDIR
import numpy as np
import pandas as pd
import glob
import os
import math
import tensorflow as tf
import keras
from keras.models import load_model 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler


class flarenet(object):
    "Creates a CNN model"

    def __init__(self,
                 #data_dir : str, 
                 #batch_size : int = 32,
                 window_size : int = 500,
                 #train : bool = True,
                 verbose: bool = False,
                 scaling: str = 'MinMax',
                 #trained_model : str = None, # filepath for trained model. 


                 #output_dir,
                 #optimizer="Adamax",
                 #loss='binary_crossentropy',
                 #metrics='accuracy'
                 ):
        
        #self.batch_size = batch_size
        self.window_size = window_size
        #self.train = train
        #self.file_list = glob.glob(f"{PACKAGEDIR}/{data_dir}/*.csv")
        self.verbose = verbose
        self.scaling = scaling
        if scaling == "MinMax":
            self.transformer = MinMaxScaler()
        elif scaling == 'Robust':
            self.transformer = RobustScaler()
        elif scaling == 'Quantile':
            self.transformer = QuantileTransformer(n_quantiles=self.window_size)
        else:
            self.transformer = None


    
    def create_training_dataset(self, 
                                data_dir : str,
                                drop_frac=0.1, 
                                
                                ):
        #data_generator = self.multiple_training_data_generator(self.file_list)
        '''self.dataset = tf.data.Dataset.from_generator(self.multiple_training_data_generator, args=[self.file_list, drop_frac],
                                        output_types = ({"inputA": tf.int32}, tf.float32),
                                        output_shapes = ({"inputA":(None,self.window_size,1)},
                                        (None,)))'''
        self.training_files = glob.glob(f"{PACKAGEDIR}/{data_dir}/*.csv")
        print(f"Found {len(self.training_files)} training files")
        self.training_dataset = tf.data.Dataset.from_generator(
            lambda: self._prepare_training_generator(self.training_files, drop_frac=drop_frac),
            output_signature=(
                {
                    "inputA": tf.TensorSpec(shape=(None, self.window_size, 1), dtype=tf.float32)
                },
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )

    def create_prediction_dataset(self,
                                  fname : str, 
                                  #data_dir : str,
                                  ):
         #glob.glob(f"{PACKAGEDIR}/{data_dir}/*.csv")
        
        print(f"Making predictions for {fname} ")
        self.prediction_dataset = tf.data.Dataset.from_generator(
            lambda: self._prepare_prediction_generator(fname),
            output_signature=(
                {
                    "inputA": tf.TensorSpec(shape=(1, self.window_size, 1), dtype=tf.float32)
                },
            )
        )
        

    def _prepare_training_generator(self, file_list, train=True, drop_frac=0.5, batch_size=32, plot=False):

        for file in file_list:
            if isinstance(file, bytes):
                file = file.decode('utf-8')
            if self.verbose:
                print(f"initilizing generator for {file}")
            
            target_data = pd.read_csv(file, index_col=False)
            
            target_lc, labels = self._prep_orbits_for_generator(target_data, 
                                                fname=file.split('/')[-1].split('.')[0],
                                                )
            
            
            lc_length = len(target_lc)
            valid_indices = np.arange(int(self.window_size/2), int(lc_length-self.window_size/2), dtype=int)
            # When training, make sure to balance the total number of flare and non-flare samples
            
            valid_labels = labels[valid_indices]
            flare_indices = valid_indices[valid_labels==1]
            nonflare_indices = valid_indices[valid_labels==0]
            n_samples =  math.floor(min(len(nonflare_indices), len(flare_indices)) * drop_frac)
            valid_flare_indices = np.random.choice(flare_indices, size=n_samples, replace=False)
            valid_nonflare_indices = np.random.choice(nonflare_indices, size=n_samples, replace=False)
            valid_indices = np.concatenate((valid_flare_indices, valid_nonflare_indices))
            np.random.shuffle(valid_indices)
            

            for j in range(0, len(valid_indices), batch_size):
                batch_indices = valid_indices[j:j+batch_size]
                bs = len(batch_indices)

                data = np.empty((bs, self.window_size, 1))
                
                label = np.empty((bs), dtype=int)

                for k, idx in enumerate(batch_indices):
                    if self.transformer != None:
                        try:
                            data[k,] = self.transformer.fit_transform(target_lc[idx-int(self.window_size/2) : idx+int(self.window_size/2)].reshape(self.window_size,1))
                        except RuntimeWarning:
                            print(target_lc[idx-int(self.window_size/2) : idx+int(self.window_size/2)])
                    else:
                        data[k,] = target_lc[idx-int(self.window_size/2) : idx+int(self.window_size/2)].reshape(self.window_size,1)
                    
                    
                    label[k] = np.asarray(labels[idx]).astype(np.int32)
                
                yield ({"inputA": data.astype(np.float32)}, label)


    def _prepare_prediction_generator(self, file):
        
        if isinstance(file, bytes):
            file = file.decode('utf-8')
        if self.verbose:
            print(f"initilizing generator for {file}")
        
        target_data = pd.read_csv(file, index_col=False)

        target_lc = self._prep_orbits_for_generator(target_data, 
                                                fname=file.split('/')[-1].split('.')[0],
                                                train=False
                                                )

        lc_length = len(target_lc)
        valid_indices = np.arange(int(self.window_size/2), int(lc_length-self.window_size/2), dtype=int)
    
        for idx in valid_indices:
            if self.transformer != None:
                data = self.transformer.fit_transform(target_lc[idx-int(self.window_size/2) : idx+int(self.window_size/2)].reshape(self.window_size,1)).reshape(1, self.window_size,1)
            else:
                data = target_lc[idx-int(self.window_size/2) : idx+int(self.window_size/2)].reshape(1, self.window_size,1)
                
            
            yield ({"inputA": data.astype(np.float32)},)
        

    # Move this over to postprocessing?
    def _prep_orbits_for_generator(self, orbit, train=True, fname=None):
        # orbit keys: ['time','flux','flux_err', 'quality','crArr','flux_with_flares','flareArr']
        #flux_list = []
        #label_list = []
        
        orbit = self._fill_gaps(orbit, train=train)
        orbit = self._pad(orbit, train=train)

        if train:
            orbit['flux_with_flares'] = orbit['flux_with_flares'].fillna(np.nanmedian(orbit['flux_with_flares'])) #flux
            flux = orbit['flux_with_flares']
            orbit['flareArr'] = orbit['flareArr'].fillna(0) #flareArr
            label = orbit['flareArr']
        else:
            orbit['normalized_flux'] = orbit['normalized_flux'].fillna(np.nanmedian(orbit['normalized_flux'])) #flux
            flux = orbit['normalized_flux']
            
        orbit['crArr'] = orbit['crArr'].fillna(0) #crArr
        #orbit['flareArr'] = orbit['flareArr'].fillna(0) #flareArr


        if fname is not None:
            #df = orbit.T
            if train:
                if not os.path.exists(f"{PACKAGEDIR}/training_data/nn_input/"):
                    os.makedirs(f"{PACKAGEDIR}/training_data/nn_input/")
                #df.save(f"{PACKAGEDIR}/training_data/nn_input/{fname}.npy", np.asarray(df))
                orbit.to_csv(f"{PACKAGEDIR}/training_data/nn_input/{fname}.csv")
            else:
                if not os.path.exists(f"{PACKAGEDIR}/prediction_data/nn_input/"):
                    os.makedirs(f"{PACKAGEDIR}/prediction_data/nn_input/")
                orbit.to_csv(f"{PACKAGEDIR}/prediction_data/nn_input/{fname}.csv")     
                self.prediction_file = f"{PACKAGEDIR}/prediction_data/nn_input/{fname}.csv"     

    
        orbit.loc[(orbit['crArr'] == 1), 'flareArr'] = 0
        #label_list.append(orbit['flareArr'].to_numpy())
        #flux_list.append(orbit['flux'].to_numpy())

        if train:
            return flux.to_numpy(dtype=np.float32), label.to_numpy(dtype=np.int32) # label_list, centroid_list, poscorr_list, cr_list
        else:
            return flux.to_numpy(dtype=np.float32)
        

    def _fill_gaps(self, orbit, train=True):
        """
        Takes in a dataframe and fills in gaps in data with sensible values
        
        Parameters
        ----------
        orbit : array
            Pandas DataFrame containing lightcurve information for one orbit
        plot : array
            Diagnostic plot to see if filled times look sensible

            
            
        Returns
        -------
        obit : array
            Updated Pandas DataFrame with rows added to fill time gaps
        """
        
        dt = 20/60/60/24 #np.nanmedian(orbit['time'][1::].values - orbit['time'][:-1:].values)

        new_times = []
        prevtime = orbit['time'].iloc[0]
        
        for t in orbit['time'][1::].values:
            if (t - prevtime) > 1.2 * dt:
                while (t - prevtime) > 1.2 * dt: # why 1.2?
                    prevtime += dt
                    new_times.append(prevtime)
                prevtime += dt

            else:
                prevtime = t

        orbit['filled'] = np.zeros(len(orbit)) # track values that are filled for later
        fill_vals = pd.DataFrame(data={'time':new_times,
                                'flux':[np.median(orbit['flux'])]*len(new_times),
                                'flux_err':[np.median(orbit['flux_err'])]*len(new_times),
                                'quality':[128]*len(new_times),
                                'crArr': np.zeros(len(new_times)),
                                'filled': np.ones(len(new_times)),
                                })
        if train:
            fill_vals['flux_with_flares'] = [np.median(orbit['flux_with_flares'])]*len(new_times)
            fill_vals['flareArr'] = np.zeros(len(new_times))
        else:
            fill_vals['normalized_flux'] = [np.median(orbit['normalized_flux'])]*len(new_times)
            fill_vals['flareArr'] = np.zeros(len(new_times))

        
        orbit = pd.concat([orbit, fill_vals]).sort_values(by='time')

        return orbit
    


    def _pad(self, orbit, window_size=500, train=True, plot=False):
        """
        Pads the beginning and end of a lightcurve.
        This makes it so that the ML model will make a prediction for all valid lightcurve points
        Parameters
        ----------
        orbit : array
            Pandas DataFrame containing lightcurve information for one orbit
        attr : array
            Array to pad
        window_size : int
            Size of sliding window for machine learning predictions.
            Pads with window_size // 2 points on each end
            
        Returns
        -------
            : array
            Padded array of values
        """

        dt = 20/60/60/24 #np.nanmedian(orbit['time'][1::].values - orbit['time'][:-1:].values)

        new_times = []
        first_time = orbit['time'].iloc[0]
        last_time = orbit['time'].iloc[-1]

        for ii in range(self.window_size // 2):
            new_times.append(first_time - dt * ii)
            new_times.append(last_time + dt * ii)
            

        fill_vals = pd.DataFrame(data={'time':new_times,
                                'flux':[np.median(orbit['flux'])]*len(new_times),
                                'flux_err':[np.median(orbit['flux_err'])]*len(new_times),
                                'quality':[128]*len(new_times),
                                'crArr': np.zeros(len(new_times)),
                                'filled': np.ones(len(new_times)),
                                })
        if train:
            fill_vals['flux_with_flares'] = [np.median(orbit['flux_with_flares'])]*len(new_times)
            fill_vals['flareArr'] = np.zeros(len(new_times))
        else:
            fill_vals['normalized_flux'] = [np.median(orbit['normalized_flux'])]*len(new_times)
            fill_vals['flareArr'] = np.zeros(len(new_times)) 
        
        orbit = pd.concat([orbit, fill_vals]).sort_values(by='time')

        return orbit


    def build_nn_model(self):
        # Define a basic ML that we can use to make sure we know the data is being imported correctly
    
        # There is 1 input light curve. In this case, we're just looking at the flux 1D time series
        inputA = keras.layers.Input(shape=(self.window_size,1), name='inputA') # flux lc

        # Convolutions on the flux lightcurve
        A = keras.layers.Conv1D(filters=32, kernel_size=21, padding="causal", activation='relu')(inputA)
        A = keras.layers.Conv1D(filters=32, kernel_size=21, padding="causal", activation='relu')(A)
        A = keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(A)
        A = keras.layers.Dropout(0.2)(A)
        # A = keras.layers.BatchNormalization()(A) # check on if these just need to be at the end
        # 2
        A = keras.layers.Conv1D(filters=64, kernel_size=7, padding="causal", activation='relu')(A)
        A = keras.layers.Conv1D(filters=64, kernel_size=7, padding="causal", activation='relu')(A)
        A = keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(A)
        A = keras.layers.Dropout(0.2)(A)

        A = keras.layers.Conv1D(filters=64, kernel_size=3, padding="causal", activation='relu')(A)
        A = keras.layers.Conv1D(filters=64, kernel_size=3, padding="causal", activation='relu')(A)
        A = keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(A)
        A = keras.layers.Dropout(0.2)(A)
        A = keras.layers.BatchNormalization()(A)


        A = keras.layers.Flatten()(A)
        A = keras.models.Model(inputA, A)


        # Combine the two convolution branches before entering the dense neural network layers
        #combined = keras.layers.concatenate([A.output, B.output, C.output, D.output])
        # combined=A.output

        # Final fully connected layers to make the prediction
        # Note that the final layer has an output shape of 1. This is because it will be a single prediction between 0 and 1
        F = keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(A.output) # look at size of input to this layer, if smaller than 512 maybe make 512 smaller. look at what stella does too
        F = keras.layers.Dropout(0.2)(F)
        F = keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal')(F)
        F = keras.layers.Dropout(0.2)(F)
        F = keras.layers.Dense(1, activation='sigmoid')(F)
        # keep the relu -> sigmoid


        multi_layer_model = keras.models.Model(inputs=(inputA), outputs=(F))
        # multi_layer_model = keras.models.Model(inputs=(inputA), outputs=(F))

        multi_layer_model.compile(
        optimizer=keras.optimizers.Adamax(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
        )
        self.model = multi_layer_model
        #return multi_layer_model
        

    def _get_nn_model(self, model : str = None):

        if model:
            print(f"Loading model {model}")
            self.model = load_model(model)
            print(model.summary())
        else: 
            print("Generating new nn model")
            self.build_nn_model()
            print(self.model.summary())
        #return model
    


    def train_model(self, 
                    #save_model : bool = True,
                    save_model_fname : str = None,
                    model : str = None, 
                    epochs : int = 20,
                    verbose : int = 1,

                    ):
        if not hasattr(self, 'training_dataset'):
            "You must create a training dataset before training the model. See self.create_training_dataset"
        else:
            self._get_nn_model(model=model)

            callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
            self.history = self.model.fit(self.training_dataset, epochs=epochs, callbacks=callback, verbose=verbose)#, use_multiprocessing=True)
            if save_model_fname != None:
                self.model.save(f"{PACKAGEDIR}/{save_model_fname}.keras")
    


    def predict_with_model(self,
                        model : str = None,
                        ):

        if not hasattr(self, 'prediction_dataset'):
            "You must create a dataset before making predictions. See self.create_prediction_dataset"
            return
        
        if isinstance(model, str):
            print(f"Loading model from {model}")
            self.model = tf.keras.models.load_model(model)
            
            preds = self.model.predict(self.prediction_dataset)
            self.preds = preds.flatten()
        
        elif model == None:
            if not hasattr(self, 'history'):
                "It doesn't look like you have a trained model available. "
            else:
                preds = self.model.predict(self.prediction_dataset)
                return preds.flatten()
            
        else:
            print("Model input not valid")
            
        
