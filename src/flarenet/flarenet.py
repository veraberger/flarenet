from . import PACKAGEDIR
from .tessprep import TessStar
from .flare_model import generate_flares
from .utils import *
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
from typing import Union
import random
from tqdm import tqdm
import traceback

#import batman


default_params = {
    'data_dir' : f"{PACKAGEDIR}/",
    'window_size':500,
    'batch_size':256,
    'batches_per_epoch':100,
    'epochs':1000,
}

def create_training_dataset( 
                    num_sources : int = 1000,
                    save_plot : bool = False,
                    num_flares : Union[int, str] = 100,
                    save_type = 'train',
                    cloud : bool = False, 
                    verbose : int = 1,
                    ):
    """
    This function can be used to generate training data by injecting flares into quiet TESS lightcurves
    
    Parameters:
    -----------
    num_sources : int = 1000
        Number of quiet stars to inject on
    save_plot : bool = False
        Save plots of the injected lightcurves
    num_flares : Union[int, str] = 100,
        Number of flares to inject for each lightcurve. 
        Default is 100, with covers ~10% of a TESS sector
    save_type :  str = 'train'
        Denotes directory to save light curves into
       Default is 'train', otherwise 'predict' or 'injection_recovery'
    cloud : bool
        Whether to use data stored by MAST on AWS. 
        If false, the files will be downloaded locally. 
    verbose : 
        Whether to print debugging information during processing


    Returns:
    -------
    None; however, lightcurves with injected flares will be saved as csv files
    """
    if save_type == 'train':
        output_dir = 'training_data/labeled_data'
    elif save_type == 'injection_recovery':
        output_dir = 'injection_recovery'
    elif save_type == 'predict':
        output_dir = 'prediction_data'
    else:
        print("save_type not recognized. Please specify 'train', 'predict', or 'injection_recovery'")
    if not os.path.exists(f"{PACKAGEDIR}/{output_dir}"):
            os.mkdir(f"{PACKAGEDIR}/{output_dir}")

    #Gets a list of quiet (non-flaring) stars
    quietstars = pd.read_csv(f"{PACKAGEDIR}/supplemental_files/ids_sectors_quietlcs.txt", sep=' ', header=0, usecols=['TIC', 'sector'])
    N=len(quietstars)
    if num_sources > N:
        raise ValueError(f"num_sources ({num_sources}) exceeds available quiet stars ({N})")

    quietstars = quietstars.iloc[:num_sources].copy()

    if verbose:
        print(f"Injecting flares into {num_sources} lightcurves.")
    

    for _, row in quietstars.iterrows():
        id = row['TIC'].astype(str)
        sector = row['sector']

        params_path = f"{PACKAGEDIR}/{output_dir}/artificial_flare_params/TIC {id}_{sector}_flareparams.npy"
        lc_path = f"{PACKAGEDIR}/{output_dir}/TIC {id}_{sector}_data.csv" 

        if os.path.exists(params_path) and os.path.exists(lc_path):
            if verbose:
                print(f"Loading existing injection for TIC {id} Sector {sector}")
            continue


        if verbose:
            print(f"Beginning processing for TIC {id} Sector {sector}")
        
        mytpf = TessStar(id, sector=sector, cloud = cloud)
        mytpf.inject_training_flares(
                        save_plot = True,
                        verbose = verbose,
                        output_dir = output_dir,
                        num_flares = num_flares,
                        )
        mytpf.save_data(save_type=save_type)

        if save_plot:
            mytpf.plot_lc()


    
    quietstars.to_csv(f"{PACKAGEDIR}/{output_dir}/{save_type}_sources.csv", index=False)

    os.system('rm -rf ~/.lksearch/cache/mastDownload/TESS/')




def split_train_val(all_files, val_fraction=0.2):
    """
    Splits data into a training and validation set for ML

    Parameters
    ----------
    all_files : list
        list of input files
    val_fraction : float, optional
        fraction of input data to use for validation, by default 0.2

    Returns
    -------
    train_files, val_files
        arrays containing files to use for training and validation
    """
    
    np.random.shuffle(all_files)
    split_idx = int(len(all_files) * (1 - val_fraction))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    return train_files, val_files

def injection_recovery(
    num_sources,
    num_flares=20,
    rise_time_minutes=1,
    save_plots=False,
    verbose : int = 1
    ):
    """
    Inject and recover synthetic flares.
    
    Parameters
    ----------
    num_sources : int
        Number of sources to test
    num_flares : int, optional
        Flares to inject per source
    rise_time_minutes : bool, optional
        How many minutes before tpeak to include when checking predictions
    save_plots : bool, optional
        Whether to save the prediction light curves
    verbose : 
        Whether to print debugging information during processing
    
    Returns:
    ----------
    - results : pandas DataFrame with columns:
        tic_id, sector, tpeak, amp, fwhm, snr, max_prediction, median_prediction
        max_prediction: max prediction within window of [tpeak - rise_time_minutes, tpeak + fwhm]
        median_prediction: median prediction within window
    """
    
    # inject flares, save light curves and a list of sources & sectors used
    create_training_dataset(
        num_sources=num_sources,
        num_flares=num_flares,
        save_type='injection_recovery',
        cloud=True, 
        save_plot=False,
        verbose=verbose
    )
    
    fn = Flarenet()
    results = []

    ij_dir = f"{PACKAGEDIR}/injection_recovery"

    # read in TIC IDs and sectors for light curves with injected flares
    sources = pd.read_csv(f"{ij_dir}/injection_recovery_sources.csv")
    sources['TIC'] = sources['TIC'].astype(str)

    if verbose:
        print(f"Completed light curve injection; beginning predictions")
    
    # run predictions; save median and max prediction per injected flare
    for _, row in sources.iterrows():
        ticid = row['TIC']
        sector = row['sector']

        df = fn.predict(
                ticid, 
                sector=sector, 
                data_dir='injection_recovery',
                save_plot=save_plots,
                verbose=verbose
            )
        
        # calculate noise level for flare SNR
        rolling_median = df['flux'].rolling(window=30, center=True).median()
        lc_std = np.nanstd(df['flux'] - rolling_median)
        
        # load injection parameters returned by create_training_dataset
        flareparams = np.load(f'{ij_dir}/artificial_flare_params/TIC {ticid}_{sector}_flareparams.npy')
        flareparams = flareparams.reshape(-1, 3)
        
        # analyze each injected flare
        for tpeak, amp, fwhm in flareparams:
            # get predictions in window around flare peak
            mask = (df['time'] > tpeak - rise_time_minutes/60/24) & (df['time'] < tpeak + fwhm)
            local_preds = df.loc[mask, 'model_prediction']
            
            results.append({
                'tic_id': ticid,
                'sector': int(sector),
                'tpeak': tpeak,
                'amp': amp,
                'fwhm': fwhm,
                'snr': amp / lc_std,
                'max_prediction': np.nanmax(local_preds),
                'median_prediction': np.nanmedian(local_preds)
            })
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{ij_dir}/injection_recovery_results.csv", index=None)
    
    return results_df


class Flarenet(object):
    "Creates a class that contains functions to run the flarenet CNN model"
    def __init__(self,
                model_weights = f"{PACKAGEDIR}/model_weights.weights.h5",
                model = f"{PACKAGEDIR}/flarenet_model.keras",
                verbose=1,
                scaling: str = 'MinMax',
                ):
        self.window_size = default_params['window_size']
        self.batch_size = default_params['batch_size']
        self.batches_per_epoch = default_params['batches_per_epoch']
        self.epochs = default_params['epochs']

        if model:
            self.model = keras.models.load_model(model)
        else:
            self.model = self.build_nn_model()

        if model_weights:
            if verbose:
                print(f"Loading model weights from file {model_weights}")
            self.model.load_weights(model_weights)
        else:
            print('Model initialized with random weights. Please train your model or input.')
        if scaling == "MinMax":
            self.transformer = MinMaxScaler()
        elif scaling == 'Robust':
            self.transformer = RobustScaler(quantile_range=(5.0, 95.0))
        elif scaling == 'Quantile':
            self.transformer = QuantileTransformer(n_quantiles=self.window_size)

    

    def prep_data(self, file, train=True, verbose=0):
        if verbose:
            print(f"Starting to prep data for file: {file}")
        try:
            if isinstance(file, str):
                target_data = pd.read_csv(file, index_col=False)
            else:
                target_data = file
            
            if verbose:
                print(f"CSV file loaded. Shape before cleaning: {target_data.shape}")
            
            target_data = target_data.sort_values('time').reset_index(drop=True)

            
            # Find gaps in data 
            dt = 20/60/60/24 #20-sec to days
            new_times = []
            prevtime = target_data['time'].iloc[0]
            for t in target_data['time'][1:]:
                if (t-prevtime) > 1.2 * dt:
                    while (t - prevtime) > 1.2 * dt:
                        prevtime += dt
                        new_times.append(prevtime)
                prevtime = t
            

            # Fill gaps in data with interpolated values. 
            # These will be masked out later, but the continuity helps predictions around gaps. 
            new_flux = np.interp(new_times, target_data['time'], target_data['flux'])
            fill_vals = pd.DataFrame(data={'time':new_times,
                'flux':new_flux + np.random.normal(loc=0.0, scale=np.nanstd(target_data['flux']), size=None),
                'flux_err': [np.median(target_data['flux_err'])] * len(new_times),
                'quality': [128] * len(new_times),
                'cr_flags': [0] * len(new_times),
                'flare_flags': [0] * len(new_times),
                'filled': [1] * len(new_times)
            })


            target_data = pd.concat([target_data, fill_vals]).sort_values('time').reset_index(drop=True)
            target_data['filled'] = target_data['filled'].fillna(0)

            if verbose:
                print(f"Gap filling complete for TIC {file}. Added {len(new_times)} new time points.")


            if verbose:
                print(f"All segments processed. Final shape: {target_data.shape}")

            start_time = target_data['time'].iloc[0]
            end_time = target_data['time'].iloc[-1]
            pad_times = [start_time - dt * (i+1) for i in range(self.window_size // 2)] + \
                    [end_time + dt * (i+1) for i in range(self.window_size // 2)]
            
            pad_vals = pd.DataFrame({
                'time': pad_times,
                'flux': [np.median(target_data['flux'])] * len(pad_times),
                'flux_err': [np.median(target_data['flux_err'])] * len(pad_times),
                'quality': [128] * len(pad_times),
                'cr_flags': [0] * len(pad_times),
                'flare_flags': [0] * len(pad_times),
                'filled': [1] * len(pad_times)
            })

            target_data = pd.concat([target_data, pad_vals]).sort_values('time').reset_index(drop=True)
            target_data = target_data.dropna(subset=['flux'])
            target_data = target_data.reset_index(drop=True)
            if verbose:
                print(f"Data after padding. Final shape: {target_data.shape}")
        

        except Exception as e:
            if verbose:
                print(f"Error during data preparation: {str(e)}")
            raise

        if verbose:
            print("Preparing final data")
        try:
            flux = target_data['flux'].astype(np.float64)
            time = target_data['time'].astype(np.float64)

            if train:
                label = target_data['flare_flags'].astype(np.int64)
                return flux.to_numpy(), label.to_numpy(), time.to_numpy()
            else: 
                return flux.to_numpy(), target_data
        except Exception as e:
            if verbose:
                print(f"Error in final data preparation: {str(e)}")
        
        if verbose:
            print("Data preparation complete")



    # Check for potential issues in the data
    def _check_data_integrity(target_data):
        print("Checking data integrity...")
        print(f"Columns present: {target_data.columns.tolist()}")
        print(f"Data types: {target_data.dtypes}")
        print(f"Missing values:\n{target_data.isnull().sum()}")
        print(f"Time range: {target_data['time'].min()} to {target_data['time'].max()}")
        print(f"Flux range: {target_data['flux'].min()} to {target_data['flux'].max()}")
        if 'flare_flags' in target_data.columns:
            print(f"Flare flag distribution: {target_data['flare_flags'].value_counts()}")

    def train_model(self, 
                    training_data_path : str = f"{PACKAGEDIR}/training_data/labeled_data/",
                    save_model : str = 'flarenet_model', # filename for model (None to not save)
                    save_weights : str = 'model_weights', # filename for model weights (None to not save)
                    verbose : int = 1,
                    ):
    
        training_files = glob.glob(f"{training_data_path}*.csv")
        if len(training_files) == 0:
            print("No training data files in given path.")
            return
    
        train_files, val_files = split_train_val(training_files)
        train_dataset = self.create_data_generator(train_files, verbose=verbose, train=True)
        val_dataset = self.create_data_generator(val_files, verbose=verbose, train=True)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
        if verbose:
            print(f"Beginning model training using {len(train_files)} training and {len(val_files)} validation lightcurves")

        history = self.model.fit(train_dataset, 
                            epochs=1, #self.epochs, #2
                            steps_per_epoch=self.batches_per_epoch,
                            callbacks=[callback], 
                            verbose=1,#always print status for training
                            validation_data=val_dataset,
                            validation_steps=self.batches_per_epoch // 5)#, use_multiprocessing=True)
        if save_model:
            self.model.save(f"{PACKAGEDIR}/{save_model}.keras")
        if save_weights:
            self.model.save_weights(f"{PACKAGEDIR}/{save_weights}.weights.h5")
        if verbose:
            print("NN model training successfully completed.")
        return history
            


    def predict(self, 
                ticid,
                data_file=None,
                sector=None,
                data_dir='prediction_data',
                verbose=1,
                save_plot= True,
                show_plot=False,
                overwrite_predictions = False,
                cloud = False,
                save_type = 'predict',
                ):
        """Predict the presence of flares in TESS 20-second data.
        TICID and sector. If no sector is specified, the first available sector for the target will be used. 

        Parameters
        ----------
        ticid : Union[int, str]:
            TESS input catalog id to make predictions on 
        sector : int, optional:
            Sector for the TESS observations. If None, the first available sector will be used. 
        data_dir : str, optional
            file path to store predictions, by default 'flarenet/prediction_data/'
        verbose : int, optional
            Whether to print out information for debugging, by default 1
        save_plot : bool, default= True
            If True, save plot showing the prediction values

        Returns
        -------
        target_df
            pandas DataFrame containing the given observations and the flarenet predictions.
        """
        # Check existing prediction data
        if not os.path.exists(f"{PACKAGEDIR}/{data_dir}"):
            os.mkdir(f"{PACKAGEDIR}/{data_dir}")

        all_data_files = glob.glob(f"{PACKAGEDIR}/{data_dir}/*.csv")


        if verbose:
            if sector == None:
                print(f"Sector not specified. Generating file for first available sector")
            else:
                print(f"Preparing file for TIC {ticid}")
        if data_file is None:
            data_file = f"{PACKAGEDIR}/{data_dir}/TIC {ticid}_{sector}_data.csv"
            
            if data_file not in all_data_files:
                ts = TessStar(ticid, sector, cloud=cloud)
                data_file = ts.save_data(save_type=save_type)

            
        if overwrite_predictions:
            completed_prediction_files = []
        else:
            if not os.path.exists(f"{PACKAGEDIR}/{data_dir}/flarenet_predictions/"):
                os.mkdir(f"{PACKAGEDIR}/{data_dir}/flarenet_predictions/")
            completed_prediction_files = os.listdir(f"{PACKAGEDIR}/{data_dir}/flarenet_predictions/")

        if f"TIC {ticid}_{sector}_predictions.csv" not in completed_prediction_files:

            try: 
                if verbose:
                    print(f"No predictions available for TIC {ticid} Sector {sector}. Starting prediction for file: {data_file}")
                
                # Prepare data using pred_data function
                target_lc, target_df = self.prep_data(data_file, train=False, verbose=verbose)

                if verbose:
                    print(f"Prepared data shape: {target_lc.shape}")

                # Create input data for the entire file
                lc_length = len(target_lc)
                preds = np.zeros(lc_length)

                windows=[]
                # Slide window through the light curve
                for i in tqdm(range(int(self.window_size/2), lc_length - int(self.window_size/2))):
                    window = target_lc[i-int(self.window_size/2):i+int(self.window_size/2)]

                    if hasattr(self, 'transformer'):
                        window = self.transformer.fit_transform(window.reshape(-1,1))
                    
                    window = window.reshape(1, self.window_size, 1)
                    windows.append(window)

                windows = np.vstack(windows)

                preds = self.model.predict(windows)

                # Pad the predictions
                preds=np.hstack((np.zeros(int(self.window_size/2)),preds[:,0],np.zeros(int(self.window_size/2))))

                if verbose:
                    print(f"Predictions shape: {preds.shape}")


                target_df['model_prediction'] = preds
                target_df = target_df[target_df['filled'] == 0]
                
                target_df = target_df.drop(columns=['filled'])#.reset_index()
                target_df.to_csv(f"{PACKAGEDIR}/{data_dir}/flarenet_predictions/TIC {ticid}_{sector}_predictions.csv", index=None)
                if verbose:
                    print(f"Successfully processed file: {data_file}")


        
            except Exception as e:
                print(f"Error processing file {data_file}: {str(e)}")
                print(traceback.format_exc())  # This will print the full traceback


        else:
            target_df = pd.read_csv(f"{PACKAGEDIR}/{data_dir}/flarenet_predictions/TIC {ticid}_{sector}_predictions.csv")

        
        fig, ax = plt.subplots(1, figsize=(12,4))
        ax.set_title(f"TIC {ticid} Sector {sector}")
        ax.scatter(target_df['time'].values, target_df['flux'].values, zorder=0, c=target_df['model_prediction'].values, s=2)
        if save_plot:
            plt.savefig(f"{PACKAGEDIR}/{data_dir}/flarenet_predictions/TIC {ticid}_{sector}_predictions.png")
        if show_plot:
            plt.show()
        plt.close()

        return target_df




    def create_data_generator(self, 
                                files,
                                drop_frac=.9,
                                verbose=1,
                                train=True,
                            ):
        """Sets up a data generator to feed the CNN model

        Parameters
        ----------
        files : list[str]
            list of files to used
        drop_frac : float, optional
            fraction of flare samples to skip during training, by default .9
        verbose : int, optional
            If 1, print status updates to screen, by default 1
        train : bool, optional
            Flag indicating if preparing a generator for training (True) or prediction (False).
            If train=True, possible false positive signals (exoplanet transits, pulsations, 
            asteroid crossings) will be randomly injected on-the-fly. By default True

        Returns
        -------
        Yields prepared data generator for use by a CNN model

        """
        
        if verbose:
            print(f"Creating generator with {len(files)} files")

        def generator():
            batch_count=0
            while True: # This creates an infinite generator
                if self.batches_per_epoch is not None and batch_count % self.batches_per_epoch == 0:
                    if verbose:
                        print(f"Starting epoch: {batch_count // self.batches_per_epoch + 1}")

                file = random.choice(files)
                if verbose:
                    print(f"Processing file: {file}")

                try:
                    if train:
                        target_lc, labels, ttime = self.prep_data(file, train=True, verbose=verbose)
                    else:
                        target_lc = self.prep_data(file, train=False, verbose=verbose)
                    if verbose:
                        print(f"File {file} processed. Shape: {target_lc.shape}")
                except Exception as e:
                    if verbose:
                        print(f"Error processing file {file}: {str(e)}")
                    continue # Skip the file if there is an error

                lc_length = len(target_lc)
                valid_indices = np.arange(int(self.window_size/2), int(lc_length - self.window_size/2), dtype=int)

                if train:
                    valid_labels = labels[valid_indices]
                    flare_indices = valid_indices[valid_labels==1]
                    nonflare_indices = valid_indices[valid_labels==0]

                    if verbose:
                        print(f"File {file}: {len(flare_indices)} flare indices, {len(nonflare_indices)} non-flare indices")
                    
                    n_samples = math.floor(min(len(nonflare_indices), len(flare_indices))*(1-drop_frac))
                    if n_samples == 0:
                        if verbose:
                            print(f"Skipping file {file}: no valid samples")
                        continue # Skip this file if there are no valid samples

                    samples_to_take = min(n_samples, self.batch_size)
                    valid_flare_indices = np.random.choice(flare_indices, size=samples_to_take, replace=False)
                    valid_nonflare_indices = np.random.choice(nonflare_indices, size=samples_to_take, replace=False)
                    valid_indices = np.concatenate((valid_flare_indices, valid_nonflare_indices))
                    np.random.shuffle(valid_indices)
                else:
                    valid_indices = valid_indices[:self.batch_size] # Take the first batch_size indices for prediction

                if verbose:
                    print(f"Taking {len(valid_indices)} samples from file {file}")
                
                batch_data = []
                batch_labels = []
                for idx in valid_indices:
                    if len(batch_data) >= self.batch_size:
                        break
                    try:
                        flux_window=target_lc[idx-int(self.window_size/2):idx+int(self.window_size/2)]
                        time_window = ttime[idx-int(self.window_size/2):idx+int(self.window_size/2)]
                        if (train) & (idx in valid_nonflare_indices):
                            rand = np.random.rand()
                            if rand < 0.2: #np.random.rand()<0.2:
                                pulsations = inject_stellar_pulsations(time_window)
                                flux_window += pulsations
                            elif rand < 0.4: #np.random.rand()<0.4:
                                transit=inject_exoplanet(time_window)
                                flux_window += transit
                            elif rand < 0.6: #np.random.rand()<0.2:
                                pulsations = inject_rr_lyrae(time_window)
                                flux_window += pulsations
                            elif rand < 0.8: #np.random.rand()< 0.1:
                                asteroid = inject_asteroid_crossing(time_window)
                                flux_window += asteroid

                        
                        if hasattr(self, 'transformer'):
                            data = self.transformer.fit_transform(flux_window.reshape(self.window_size,1))
                        else:
                            data = flux_window.reshape(self.window_size,1)
                        if not np.any(np.isnan(data)):
                            batch_data.append(data)
                            if train:
                                batch_labels.append(labels[idx])
                    except Exception as e:
                        if verbose:
                            print(f"Error processing sample at index {idx} from file {file}: {str(e)}")
                        continue
                if verbose:
                    print(f"Current batch size: {len(batch_data)}")
                batch_count += 1
                if self.batches_per_epoch is not None and batch_count % self.batches_per_epoch == 0:
                    if verbose:
                        print(f"Epooch {batch_count // self.batches_per_epoch} completed")
                
                # Yield the batch
                if train:
                    yield ({"inputA": np.array(batch_data).astype(np.float64)}, np.array(batch_labels))
                else:
                    yield {"inputA": np.array(batch_data).astype(np.float64)}
        
        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                {
                    "inputA": tf.TensorSpec(shape=(None, self.window_size, 1), dtype=tf.float64)
                },
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            ) if train else (
                {
                    "inputA": tf.TensorSpec(shape=(None, self.window_size, 1), dtype=tf.float64)
                }
            )
        )
    
    def build_nn_model(self):
        inputA = tf.keras.layers.Input(shape=(self.window_size,1), name='inputA')
    
        A = tf.keras.layers.Conv1D(filters=16, kernel_size=24, padding="same", activation='leaky_relu')(inputA)
        A = tf.keras.layers.Dropout(0.5)(A)
        A = tf.keras.layers.Conv1D(filters=16, kernel_size=24, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.BatchNormalization()(A)
        A = tf.keras.layers.Conv1D(filters=16, kernel_size=24, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.Dropout(0.5)(A)
        A = tf.keras.layers.Conv1D(filters=16, kernel_size=24, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.BatchNormalization()(A)
        A = tf.keras.layers.Conv1D(filters=16, kernel_size=24, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.Dropout(0.5)(A)
        A = tf.keras.layers.Conv1D(filters=16, kernel_size=24, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.BatchNormalization()(A)
    
    
        A = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(A)
        A = tf.keras.layers.Conv1D(filters=32, kernel_size=12, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.Dropout(0.5)(A)
        A = tf.keras.layers.Conv1D(filters=32, kernel_size=12, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.BatchNormalization()(A)
        A = tf.keras.layers.Conv1D(filters=32, kernel_size=12, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.Dropout(0.5)(A)
        A = tf.keras.layers.Conv1D(filters=32, kernel_size=12, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.BatchNormalization()(A)
        A = tf.keras.layers.Conv1D(filters=32, kernel_size=12, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.Dropout(0.5)(A)
        A = tf.keras.layers.Conv1D(filters=32, kernel_size=12, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.BatchNormalization()(A)
    
    
        A = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(A)
        A = tf.keras.layers.Conv1D(filters=64, kernel_size=6, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.Dropout(0.5)(A)
        A = tf.keras.layers.Conv1D(filters=64, kernel_size=6, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.BatchNormalization()(A)
        A = tf.keras.layers.Conv1D(filters=64, kernel_size=6, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.Dropout(0.5)(A)
        A = tf.keras.layers.Conv1D(filters=64, kernel_size=6, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.BatchNormalization()(A)
        A = tf.keras.layers.Conv1D(filters=64, kernel_size=6, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.Dropout(0.5)(A)
        A = tf.keras.layers.Conv1D(filters=64, kernel_size=6, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.BatchNormalization()(A)
    
        A = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(A)
        A = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.Dropout(0.5)(A)
        A = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.BatchNormalization()(A)
        A = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.Dropout(0.5)(A)
        A = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.BatchNormalization()(A)
        A = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.Dropout(0.5)(A)
        A = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.BatchNormalization()(A)
    
        A = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(A)
        A = tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.Dropout(0.5)(A)
        A = tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.BatchNormalization()(A)
        A = tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.Dropout(0.5)(A)
        A = tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.BatchNormalization()(A)
        A = tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.Dropout(0.5)(A)
        A = tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding="same", activation='leaky_relu')(A)
        A = tf.keras.layers.BatchNormalization()(A)
    
    
        A = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(A)
        A = tf.keras.layers.Flatten()(A)
        F = tf.keras.layers.Dense(1024, activation='leaky_relu', kernel_initializer='he_normal')(A)
        F = tf.keras.layers.Dropout(0.5)(F)
        F = tf.keras.layers.Dense(512, activation='leaky_relu', kernel_initializer='he_normal')(F)
        F = tf.keras.layers.Dropout(0.5)(F)
        F = tf.keras.layers.Dense(1, activation='sigmoid')(F)
    
        model = tf.keras.models.Model(inputs=(inputA), outputs=(F))
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model