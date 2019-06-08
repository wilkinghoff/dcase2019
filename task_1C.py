import pandas as pd
import numpy as np
import librosa
import keras
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser
import os
from keras import backend as K
from sklearn.linear_model import LogisticRegression
from scipy.stats.mstats import gmean


class LogMelExtractor(object):
    """
    Original source code (before changes): https://github.com/qiuqiangkong/dcase2019_task1
    """
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        """Log mel feature extractor.

        Args:
          sample_rate: int
          window_size: int
          hop_size: int
          mel_bins: int
          fmin: int, minimum frequency of mel filter banks
          fmax: int, maximum frequency of mel filter banks
        """

        self.window_size = window_size
        self.hop_size = hop_size
        self.window_func = np.hanning(window_size)

        self.melW = librosa.filters.mel(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax).T

    def transform(self, audio):
        """Extract feature of a singlechannel audio file.

        Args:
          audio: (samples,)

        Returns:
          feature: (frames_num, freq_bins)
        """

        window_size = self.window_size
        hop_size = self.hop_size
        window_func = self.window_func

        # Compute short-time Fourier transform
        stft_matrix = librosa.core.stft(
            y=audio,
            n_fft=window_size,
            hop_length=hop_size,
            window=window_func,
            center=True,
            dtype=np.complex64,
            pad_mode='reflect').T
        '''(N, n_fft // 2 + 1)'''

        # Mel spectrogram
        mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, self.melW)

        # Log mel spectrogram
        logmel_spectrogram = librosa.core.power_to_db(
            mel_spectrogram, ref=1.0, amin=1e-10,
            top_db=None)

        logmel_spectrogram = logmel_spectrogram.astype(np.float32).transpose()

        # harmonic and percussive parts
        D_harmonic, D_percussive = librosa.decompose.hpss(mel_spectrogram)

        log_D_harmonic = librosa.core.power_to_db(
            D_harmonic, ref=1.0, amin=1e-10,
            top_db=None)

        log_D_percussive = librosa.core.power_to_db(
            D_percussive, ref=1.0, amin=1e-10,
            top_db=None)

        log_D_harmonic = log_D_harmonic.astype(np.float32).transpose()
        log_D_percussive = log_D_percussive.astype(np.float32).transpose()
        return logmel_spectrogram, log_D_harmonic, log_D_percussive


def model_cnn(feat_length, nceps):
    """
    Definition of the 2D CNN used for mel spectrograms.
    :param feat_length: length of the features being used
    :param nceps: Number of mel filters.
    :param trainable: Should the layers be trainable or constant
    :param name: Name of the model , important when reloading the weights
    :return: CNN model
    """
    input = keras.layers.Input(shape=(nceps, feat_length, 1), dtype='float32')

    x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 3))(x)

    x = keras.layers.Conv2D(128, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2D(128, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 3))(x)

    x = keras.layers.Conv2D(196, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2D(196, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 3))(x)

    x = keras.layers.Conv2D(256, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2D(256, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    model_output = keras.layers.Dense(10, activation='softmax')(x)
    return input, model_output


def model_dcae_def(feat_length, nceps):
    """
    Definition of the 2D DCAE used for mel spectrograms.
    :param feat_length: length of the features being used
    :param nceps: Number of mel filters.
    :param trainable: Should the layers be trainable or constant
    :param name: Name of the model , important when reloading the weights
    :return: CNN model
    """
    input = keras.layers.Input(shape=(nceps, feat_length, 1), dtype='float32')

    # encoder (same as discriminative) CNN model
    x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)

    x = keras.layers.Conv2D(128, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2D(128, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)

    # decoder
    x = keras.layers.Conv2D(128, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2D(128, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.UpSampling2D(size=(2, 2))(x)
    x = keras.layers.ZeroPadding2D(padding=((0, 0), (0, 1)))(x)

    x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.UpSampling2D(size=(2, 2))(x)

    model_output = keras.layers.Conv2D(1, kernel_size=3, padding='same', activation='relu')(x)
    return input, model_output



########################################################################################################################
# Load data and compute spectrograms
########################################################################################################################

nceps_mel_spec = 64
background_subtraction_kernel_size = (43, 11)
# load train data
print('Loading train data')
dataset_path = 'D:/dcase2019_task1_baseline-master/datasets/TAU-urban-acoustic-scenes-2019-openset-development/'
train_sheet = pd.DataFrame({'File': [l.split("\t")[0].split("\n")[0] for l in open(dataset_path + 'evaluation_setup/fold1_train.csv','r').readlines()[1:]],
                          'Category': [l.split("\t")[1].split("\n")[0] for l in open(dataset_path + 'evaluation_setup/fold1_train.csv','r').readlines()[1:]]})
categories = np.unique(train_sheet.Category)
extractor = LogMelExtractor(32000, 1024, 500, 64, 50, 16000)
if os.path.isfile('train_labels.npy') and os.path.isfile('train_mel_spec_feats' + str(nceps_mel_spec) + '.npy')\
        and os.path.isfile('train_harmonic_feats' + str(nceps_mel_spec) + '.npy')\
        and os.path.isfile('train_percussive_feats' + str(nceps_mel_spec) + '.npy'):
    train_labels = np.load('train_labels.npy')
    train_mel_spec_feats = np.load('train_mel_spec_feats' + str(nceps_mel_spec) + '.npy')
    train_harmonic_feats = np.load('train_harmonic_feats' + str(nceps_mel_spec) + '.npy')
    train_percussive_feats = np.load('train_percussive_feats' + str(nceps_mel_spec) + '.npy')
else:
    train_mel_spec_feats = []
    train_harmonic_feats = []
    train_percussive_feats = []
    train_labels = []
    for label, category in enumerate(categories):
        print(category)
        if category != 'unknown':
            for count, file in enumerate(train_sheet.File[train_sheet.Category == category]):
                print('Processing file ' + str(count))
                file_path = dataset_path + file
                wav, fs = librosa.load(file_path)
                wav = librosa.util.normalize(wav)
                mel_spec, D_harmonic, D_percussive = extractor.transform(wav)
                train_mel_spec_feats.append(mel_spec)
                train_harmonic_feats.append(D_harmonic)
                train_percussive_feats.append(D_percussive)
                train_labels.append(label)
    # reshape arrays and store
    train_labels = np.array(train_labels)
    train_mel_spec_feats = np.expand_dims(np.array(train_mel_spec_feats, dtype=np.float32), axis=-1)
    train_harmonic_feats = np.expand_dims(np.array(train_harmonic_feats, dtype=np.float32), axis=-1)
    train_percussive_feats = np.expand_dims(np.array(train_percussive_feats, dtype=np.float32), axis=-1)
    np.save('train_labels.npy', train_labels)
    np.save('train_mel_spec_feats' + str(nceps_mel_spec) + '.npy', train_mel_spec_feats)
    np.save('train_harmonic_feats' + str(nceps_mel_spec) + '.npy', train_harmonic_feats)
    np.save('train_percussive_feats' + str(nceps_mel_spec) + '.npy', train_percussive_feats)

# load evaluation data
print('Loading evaluation data')
if os.path.isfile('eval_labels.npy') and os.path.isfile('eval_mel_spec_feats' + str(nceps_mel_spec) + '.npy')\
        and os.path.isfile('eval_harmonic_feats' + str(nceps_mel_spec) + '.npy')\
        and os.path.isfile('eval_percussive_feats' + str(nceps_mel_spec) + '.npy'):
    eval_labels = np.load('eval_labels.npy')
    eval_mel_spec_feats = np.load('eval_mel_spec_feats' + str(nceps_mel_spec) + '.npy')
    eval_harmonic_feats = np.load('eval_harmonic_feats' + str(nceps_mel_spec) + '.npy')
    eval_percussive_feats = np.load('eval_percussive_feats' + str(nceps_mel_spec) + '.npy')
else:
    eval_sheet = pd.DataFrame({'File': [l.split("\t")[0].split("\n")[0] for l in open(dataset_path + 'evaluation_setup/fold1_evaluate.csv','r').readlines()[1:]],
                               'Category': [l.split("\t")[1].split("\n")[0] for l in open(dataset_path + 'evaluation_setup/fold1_evaluate.csv','r').readlines()[1:]]})
    eval_mel_spec_feats = []
    eval_harmonic_feats = []
    eval_percussive_feats = []
    eval_labels = []
    for label, category in enumerate(categories):
        print(category)
        if category != 'unknown':
            for count, file in enumerate(eval_sheet.File[eval_sheet.Category == category]):
                print('Processing file ' + str(count))
                file_path = dataset_path + file
                wav, fs = librosa.load(file_path)
                wav = librosa.util.normalize(wav)
                mel_spec, D_harmonic, D_percussive = extractor.transform(wav)
                eval_mel_spec_feats.append(mel_spec)
                eval_harmonic_feats.append(D_harmonic)
                eval_percussive_feats.append(D_percussive)
                eval_labels.append(label)

    # reshape arrays and store
    eval_labels = np.array(eval_labels)
    eval_mel_spec_feats = np.expand_dims(np.array(eval_mel_spec_feats, dtype=np.float32), axis=-1)
    eval_harmonic_feats = np.expand_dims(np.array(eval_harmonic_feats, dtype=np.float32), axis=-1)
    eval_percussive_feats = np.expand_dims(np.array(eval_percussive_feats, dtype=np.float32), axis=-1)
    np.save('eval_labels.npy', eval_labels)
    np.save('eval_mel_spec_feats' + str(nceps_mel_spec) + '.npy', eval_mel_spec_feats)
    np.save('eval_harmonic_feats' + str(nceps_mel_spec) + '.npy', eval_harmonic_feats)
    np.save('eval_percussive_feats' + str(nceps_mel_spec) + '.npy', eval_percussive_feats)

# load unknown samples
print('Loading unknown data')
if os.path.isfile('unknown_mel_spec_feats' + str(nceps_mel_spec) + '.npy')\
        and os.path.isfile('unknown_harmonic_feats' + str(nceps_mel_spec) + '.npy')\
        and os.path.isfile('unknown_percussive_feats' + str(nceps_mel_spec) + '.npy'):
    unknown_mel_spec_feats = np.load('unknown_mel_spec_feats' + str(nceps_mel_spec) + '.npy')
    unknown_harmonic_feats = np.load('unknown_harmonic_feats' + str(nceps_mel_spec) + '.npy')
    unknown_percussive_feats = np.load('unknown_percussive_feats' + str(nceps_mel_spec) + '.npy')
else:
    eval_sheet = pd.DataFrame({'File': [l.split("\t")[0].split("\n")[0] for l in open(dataset_path + 'evaluation_setup/fold1_evaluate.csv','r').readlines()[1:]],
                               'Category': [l.split("\t")[1].split("\n")[0] for l in open(dataset_path + 'evaluation_setup/fold1_evaluate.csv','r').readlines()[1:]]})
    unknown_mel_spec_feats = []
    unknown_harmonic_feats = []
    unknown_percussive_feats = []
    category = 'unknown'
    for count, file in enumerate(train_sheet.File[train_sheet.Category == category]):
        print('Processing train file ' + str(count))
        file_path = dataset_path + file
        wav, fs = librosa.load(file_path)
        wav = librosa.util.normalize(wav)
        mel_spec, D_harmonic, D_percussive = extractor.transform(wav)
        unknown_mel_spec_feats.append(mel_spec)
        unknown_harmonic_feats.append(D_harmonic)
        unknown_percussive_feats.append(D_percussive)
    for count, file in enumerate(eval_sheet.File[eval_sheet.Category == category]):
        print('Processing eval file ' + str(count))
        file_path = dataset_path + file
        wav, fs = librosa.load(file_path)
        wav = librosa.util.normalize(wav)
        mel_spec, D_harmonic, D_percussive = extractor.transform(wav)
        unknown_mel_spec_feats.append(mel_spec)
        unknown_harmonic_feats.append(D_harmonic)
        unknown_percussive_feats.append(D_percussive)

    # reshape arrays and store
    unknown_mel_spec_feats = np.expand_dims(np.array(unknown_mel_spec_feats, dtype=np.float32), axis=-1)
    unknown_harmonic_feats = np.expand_dims(np.array(unknown_harmonic_feats, dtype=np.float32), axis=-1)
    unknown_percussive_feats = np.expand_dims(np.array(unknown_percussive_feats, dtype=np.float32), axis=-1)
    np.save('unknown_mel_spec_feats' + str(nceps_mel_spec) + '.npy', unknown_mel_spec_feats)
    np.save('unknown_harmonic_feats' + str(nceps_mel_spec) + '.npy', unknown_harmonic_feats)
    np.save('unknown_percussive_feats' + str(nceps_mel_spec) + '.npy', unknown_percussive_feats)

# load leaderboard data
print('Loading leaderboard data')
dataset_leaderboard_path = 'D:/dcase2019_task1_baseline-master/datasets/TAU-urban-acoustic-scenes-2019-openset-leaderboard/'
leaderboard_sheet = pd.DataFrame({'File': [l.split("\t")[0].split("\n")[0] for l in open(dataset_leaderboard_path + 'evaluation_setup/test.csv','r').readlines()[1:]]})
if os.path.isfile('leaderboard_mel_spec_feats' + str(nceps_mel_spec) + '.npy')\
        and os.path.isfile('leaderboard_harmonic_feats' + str(nceps_mel_spec) + '.npy')\
        and os.path.isfile('leaderboard_percussive_feats' + str(nceps_mel_spec) + '.npy'):
    leaderboard_mel_spec_feats = np.load('leaderboard_mel_spec_feats' + str(nceps_mel_spec) + '.npy')
    leaderboard_harmonic_feats = np.load('leaderboard_harmonic_feats' + str(nceps_mel_spec) + '.npy')
    leaderboard_percussive_feats = np.load('leaderboard_percussive_feats' + str(nceps_mel_spec) + '.npy')
else:
    leaderboard_mel_spec_feats = []
    leaderboard_harmonic_feats = []
    leaderboard_percussive_feats = []
    for count, file in enumerate(leaderboard_sheet.File):
        print('Processing file ' + str(count))
        file_path = dataset_leaderboard_path + file
        wav, fs = librosa.load(file_path)
        wav = librosa.util.normalize(wav)
        mel_spec, D_harmonic, D_percussive = extractor.transform(wav)
        leaderboard_mel_spec_feats.append(mel_spec)
        leaderboard_harmonic_feats.append(D_harmonic)
        leaderboard_percussive_feats.append(D_percussive)
    # reshape arrays and store
    leaderboard_mel_spec_feats = np.expand_dims(np.array(leaderboard_mel_spec_feats, dtype=np.float32), axis=-1)
    leaderboard_harmonic_feats = np.expand_dims(np.array(leaderboard_harmonic_feats, dtype=np.float32), axis=-1)
    leaderboard_percussive_feats = np.expand_dims(np.array(leaderboard_percussive_feats, dtype=np.float32), axis=-1)
    np.save('leaderboard_mel_spec_feats' + str(nceps_mel_spec) + '.npy', leaderboard_mel_spec_feats)
    np.save('leaderboard_harmonic_feats' + str(nceps_mel_spec) + '.npy', leaderboard_harmonic_feats)
    np.save('leaderboard_percussive_feats' + str(nceps_mel_spec) + '.npy', leaderboard_percussive_feats)

# load test data
print('Loading test data')
dataset_test_path = 'D:/dcase2019_task1_baseline-master/datasets/TAU-urban-acoustic-scenes-2019-openset-evaluation/'
test_sheet = pd.DataFrame({'File': [l.split("\t")[0].split("\n")[0] for l in open(dataset_test_path + 'evaluation_setup/test.csv','r').readlines()[1:]]})
if os.path.isfile('test_mel_spec_feats' + str(nceps_mel_spec) + '.npy')\
        and os.path.isfile('test_harmonic_feats' + str(nceps_mel_spec) + '.npy')\
        and os.path.isfile('test_percussive_feats' + str(nceps_mel_spec) + '.npy'):
    test_mel_spec_feats = np.load('test_mel_spec_feats' + str(nceps_mel_spec) + '.npy')
    test_harmonic_feats = np.load('test_harmonic_feats' + str(nceps_mel_spec) + '.npy')
    test_percussive_feats = np.load('test_percussive_feats' + str(nceps_mel_spec) + '.npy')
else:
    test_mel_spec_feats = []
    test_harmonic_feats = []
    test_percussive_feats = []
    for count, file in enumerate(test_sheet.File):
        print('Processing file ' + str(count))
        file_path = dataset_test_path + file
        wav, fs = librosa.load(file_path)
        wav = librosa.util.normalize(wav)
        mel_spec, D_harmonic, D_percussive = extractor.transform(wav)
        test_mel_spec_feats.append(mel_spec)
        test_harmonic_feats.append(D_harmonic)
        test_percussive_feats.append(D_percussive)
    # reshape arrays and store
    test_mel_spec_feats = np.expand_dims(np.array(test_mel_spec_feats, dtype=np.float32), axis=-1)
    test_harmonic_feats = np.expand_dims(np.array(test_harmonic_feats, dtype=np.float32), axis=-1)
    test_percussive_feats = np.expand_dims(np.array(test_percussive_feats, dtype=np.float32), axis=-1)
    np.save('test_mel_spec_feats' + str(nceps_mel_spec) + '.npy', test_mel_spec_feats)
    np.save('test_harmonic_feats' + str(nceps_mel_spec) + '.npy', test_harmonic_feats)
    np.save('test_percussive_feats' + str(nceps_mel_spec) + '.npy', test_percussive_feats)

########################################################################################################################
# train and evaluate DCAEs
########################################################################################################################

batch_size = 32
batch_size_test = 32
epochs = 100
aeons = 10
y_train_cat = keras.utils.np_utils.to_categorical(train_labels, num_classes=len(categories)-1)
y_eval_cat = keras.utils.np_utils.to_categorical(eval_labels, num_classes=len(categories)-1)

feature_types = ['mel_spec', 'harmonic', 'percussive']
for k_feat, feature_type in enumerate(feature_types):
    outlier_scores_clf = np.zeros((3, leaderboard_mel_spec_feats.shape[0]))
    outlier_scores_eval_clf = np.zeros((3, eval_mel_spec_feats.shape[0]))
    outlier_scores_unknown_clf = np.zeros((3, unknown_mel_spec_feats.shape[0]))
    outlier_scores_test_clf = np.zeros((3, test_mel_spec_feats.shape[0]))
    if feature_type == 'mel_spec':
        train_data = train_mel_spec_feats
        eval_data = eval_mel_spec_feats
        unknown_data = unknown_mel_spec_feats
        leaderboard_data = leaderboard_mel_spec_feats
        test_data = test_mel_spec_feats
    elif feature_type == 'harmonic':
        train_data = train_harmonic_feats
        eval_data = eval_harmonic_feats
        unknown_data = unknown_harmonic_feats
        leaderboard_data = leaderboard_harmonic_feats
        test_data = test_harmonic_feats
    elif feature_type == 'percussive':
        train_data = train_percussive_feats
        eval_data = eval_percussive_feats
        unknown_data = unknown_percussive_feats
        leaderboard_data = leaderboard_percussive_feats
        test_data = test_percussive_feats
    else:
        raise ValueError('Invalid feature type!')
    outlier_scores = np.zeros((leaderboard_data.shape[0], len(categories)-1))
    outlier_scores_eval = np.zeros((eval_data.shape[0], len(categories)-1))
    outlier_scores_unknown = np.zeros((unknown_data.shape[0], len(categories)-1))
    outlier_scores_test = np.zeros((test_data.shape[0], len(categories)-1))

    for j in np.arange(len(categories)):
        K.clear_session()
        category = categories[j]
        if category != 'unknown':
            print('Training/evaluating DCAE for category ' + category)
            relevant_idcs = [label == j for label in train_labels]
            train_data_dcae = np.copy(train_data[relevant_idcs])
            leaderboard_data_dcae = np.copy(leaderboard_data)
            unknown_data_dcae = np.copy(unknown_data)
            eval_data_dcae = np.copy(eval_data)
            test_data_dcae = np.copy(test_data)

            # feature normalization
            print('Normalizing data with class specific characteristics')
            eps = 1e-12
            mean = np.mean(train_data_dcae, axis=0)
            std = np.std(train_data_dcae, axis=0)
            for k in np.arange(train_data_dcae.shape[0]):
                train_data_dcae[k] = (train_data_dcae[k] - mean) / (std + eps)
            for k in np.arange(eval_data_dcae.shape[0]):
                eval_data_dcae[k] = (eval_data_dcae[k] - mean) / (std + eps)
            for k in np.arange(unknown_data_dcae.shape[0]):
                unknown_data_dcae[k] = (unknown_data_dcae[k] - mean) / (std + eps)
            for k in np.arange(leaderboard_data_dcae.shape[0]):
                leaderboard_data_dcae[k] = (leaderboard_data_dcae[k] - mean) / (std + eps)
            for k in np.arange(test_data_dcae.shape[0]):
                test_data_dcae[k] = (test_data_dcae[k] - mean) / (std + eps)

            # compile model
            input_dcae, model_output_dcae = model_dcae_def(feat_length=442, nceps=nceps_mel_spec)
            model_dcae = keras.Model(inputs=[input_dcae], outputs=[model_output_dcae])
            model_dcae.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.adam(decay=0.0001))

            for k in np.arange(aeons):
                # fit model
                weight_path = 'wts_dcae_' + feature_type + '_' + category + '_' + str(k+1) + 'k.h5'
                if not os.path.isfile(weight_path):
                    model_dcae.fit(train_data_dcae, train_data_dcae, verbose=2, batch_size=batch_size, epochs=epochs)
                    model_dcae.save(weight_path)
                else:
                    model_dcae = keras.models.load_model(weight_path)

            # predict on eval data
            print('Computing loss for eval data')
            for k in np.arange(eval_data_dcae.shape[0]):
                feat = np.expand_dims(eval_data_dcae[k], axis=0)
                outlier_scores_eval[k, j] = model_dcae.evaluate(feat, feat, verbose=0)

            # predict on unknown data
            print('Computing loss for unknown data')
            for k in np.arange(unknown_data_dcae.shape[0]):
                feat = np.expand_dims(unknown_data_dcae[k], axis=0)
                outlier_scores_unknown[k, j] = model_dcae.evaluate(feat, feat, verbose=0)

            # predict on leaderboard data
            print('Computing loss for leaderboard data')
            for k in np.arange(leaderboard_data_dcae.shape[0]):
                feat = np.expand_dims(leaderboard_data_dcae[k], axis=0)
                outlier_scores[k, j] = model_dcae.evaluate(feat, feat, verbose=0)

            # predict on test data
            print('Computing loss for test data')
            for k in np.arange(test_data_dcae.shape[0]):
                feat = np.expand_dims(test_data_dcae[k], axis=0)
                outlier_scores_test[k, j] = model_dcae.evaluate(feat, feat, verbose=0)

    # train logistic regression model to transform scores
    print('Estimating scores via logistic regression')
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(np.concatenate([outlier_scores_eval, outlier_scores_unknown], axis=0),
            np.concatenate([np.ones(outlier_scores_eval.shape[0]), np.zeros(outlier_scores_unknown.shape[0])], axis=-1))
    outlier_scores_clf[k_feat] = clf.predict_proba(outlier_scores)[:, 1]
    outlier_scores_unknown_clf[k_feat] = clf.predict_proba(outlier_scores_unknown)[:, 1]
    outlier_scores_eval_clf[k_feat] = clf.predict_proba(outlier_scores_eval)[:, 1]
    outlier_scores_test_clf[k_feat] = clf.predict_proba(outlier_scores_test)[:, 1]

np.save('outlier_scores_eval.npy', outlier_scores_eval)
np.save('outlier_scores_unknown.npy', outlier_scores_unknown)
np.save('outlier_scores.npy', outlier_scores)
np.save('outlier_scores_eval_clf.npy', outlier_scores_eval_clf)
np.save('outlier_scores_unknown_clf.npy', outlier_scores_unknown_clf)
np.save('outlier_scores_clf.npy', outlier_scores_clf)

########################################################################################################################
# Preprocessing for CNNs
########################################################################################################################
train_labels = np.concatenate([train_labels, eval_labels], axis=0)
train_mel_spec_feats = np.concatenate([train_mel_spec_feats, eval_mel_spec_feats], axis=0)
train_harmonic_feats = np.concatenate([train_harmonic_feats, eval_harmonic_feats], axis=0)
train_percussive_feats = np.concatenate([train_percussive_feats, eval_percussive_feats], axis=0)

# feature normalization
print('Normalizing data globally')
eps = 1e-12
mean_mel_spec = np.mean(train_mel_spec_feats, axis=0)
std_mel_spec = np.std(train_mel_spec_feats, axis=0)
for k in np.arange(train_mel_spec_feats.shape[0]):
    train_mel_spec_feats[k] = (train_mel_spec_feats[k]-mean_mel_spec)/(std_mel_spec+eps)
for k in np.arange(eval_mel_spec_feats.shape[0]):
    eval_mel_spec_feats[k] = (eval_mel_spec_feats[k]-mean_mel_spec)/(std_mel_spec+eps)
for k in np.arange(unknown_mel_spec_feats.shape[0]):
    unknown_mel_spec_feats[k] = (unknown_mel_spec_feats[k]-mean_mel_spec)/(std_mel_spec+eps)
for k in np.arange(leaderboard_mel_spec_feats.shape[0]):
    leaderboard_mel_spec_feats[k] = (leaderboard_mel_spec_feats[k]-mean_mel_spec)/(std_mel_spec+eps)
for k in np.arange(test_mel_spec_feats.shape[0]):
    test_mel_spec_feats[k] = (test_mel_spec_feats[k]-mean_mel_spec)/(std_mel_spec+eps)
mean_harmonic = np.mean(train_harmonic_feats, axis=0)
std_harmonic = np.std(train_harmonic_feats, axis=0)
for k in np.arange(train_harmonic_feats.shape[0]):
    train_harmonic_feats[k] = (train_harmonic_feats[k]-mean_harmonic)/(std_harmonic+eps)
for k in np.arange(eval_harmonic_feats.shape[0]):
    eval_harmonic_feats[k] = (eval_harmonic_feats[k]-mean_harmonic)/(std_harmonic+eps)
for k in np.arange(unknown_harmonic_feats.shape[0]):
    unknown_harmonic_feats[k] = (unknown_harmonic_feats[k]-mean_harmonic)/(std_harmonic+eps)
for k in np.arange(leaderboard_harmonic_feats.shape[0]):
    leaderboard_harmonic_feats[k] = (leaderboard_harmonic_feats[k]-mean_harmonic)/(std_harmonic+eps)
for k in np.arange(test_harmonic_feats.shape[0]):
    test_harmonic_feats[k] = (test_harmonic_feats[k]-mean_harmonic)/(std_harmonic+eps)
mean_percussive = np.mean(train_percussive_feats, axis=0)
std_percussive = np.std(train_percussive_feats, axis=0)
for k in np.arange(train_percussive_feats.shape[0]):
    train_percussive_feats[k] = (train_percussive_feats[k]-mean_percussive)/(std_percussive+eps)
for k in np.arange(eval_percussive_feats.shape[0]):
    eval_percussive_feats[k] = (eval_percussive_feats[k]-mean_percussive)/(std_percussive+eps)
for k in np.arange(unknown_percussive_feats.shape[0]):
    unknown_percussive_feats[k] = (unknown_percussive_feats[k]-mean_percussive)/(std_percussive+eps)
for k in np.arange(leaderboard_percussive_feats.shape[0]):
    leaderboard_percussive_feats[k] = (leaderboard_percussive_feats[k]-mean_percussive)/(std_percussive+eps)
for k in np.arange(test_percussive_feats.shape[0]):
    test_percussive_feats[k] = (test_percussive_feats[k]-mean_percussive)/(std_percussive+eps)

########################################################################################################################
# train base cnn on all data
########################################################################################################################
batch_size = 32
batch_size_test = 32
epochs = 1000
aeons = 6
alpha = 1
y_train_cat = keras.utils.np_utils.to_categorical(train_labels, num_classes=len(categories)-1)
y_eval_cat = keras.utils.np_utils.to_categorical(eval_labels, num_classes=len(categories)-1)

feature_types = ['mel_spec', 'harmonic', 'percussive']
pred_leaderboard_base = np.ones((3, leaderboard_mel_spec_feats.shape[0], len(categories)))
pred_test_base = np.ones((3, test_mel_spec_feats.shape[0], len(categories)))
for k_feat, feature_type in enumerate(feature_types):
    K.clear_session()
    if feature_type == 'mel_spec':
        train_data = train_mel_spec_feats
        eval_data = eval_mel_spec_feats
    elif feature_type == 'harmonic':
        train_data = train_harmonic_feats
        eval_data = eval_harmonic_feats
    elif feature_type == 'percussive':
        train_data = train_percussive_feats
        eval_data = eval_percussive_feats
    else:
        raise ValueError('Invalid feature type!')

    # compile model
    input, model_output = model_cnn(442, nceps=nceps_mel_spec)
    model_base = keras.Model(inputs=[input], outputs=[model_output])
    model_base.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(decay=0.0001),
                           metrics=['accuracy'])

    # create data generator for mixup and random erasing for every batch
    datagen_mel_spec = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.6,  # up to 60 percent shift in time
        height_shift_range=3,  # up to 3 mel bins up or down
        preprocessing_function=get_random_eraser(v_l=np.min(train_data), v_h=np.max(train_data))
        )
    training_generator_mel_spec = MixupGenerator(train_data, y_train_cat, batch_size=batch_size,
                                                 alpha=alpha, datagen=datagen_mel_spec)

    for k in np.arange(aeons):
        # fit model
        weight_path = 'wts_' + feature_type + '_' + str(k+1) + 'k.h5'
        if not os.path.isfile(weight_path):
            model_base.fit_generator(generator=training_generator_mel_spec(), verbose=2,
                                     steps_per_epoch=train_data.shape[0] // batch_size, epochs=epochs)
            model_base.save(weight_path)
        else:
            model_base = keras.models.load_model(weight_path)

    # predict
    print('Predicting on leaderboard data')
    if feature_type == 'mel_spec':
        pred_leaderboard_base[k_feat] = model_base.predict(leaderboard_mel_spec_feats)
    elif feature_type == 'harmonic':
        pred_leaderboard_base[k_feat] = model_base.predict(leaderboard_harmonic_feats)
    elif feature_type == 'percussive':
        pred_leaderboard_base[k_feat] = model_base.predict(leaderboard_percussive_feats)
    else:
        raise ValueError('Invalid feature type!')

    # predict
    print('Predicting on test data')
    if feature_type == 'mel_spec':
        pred_test_base[k_feat] = model_base.predict(test_mel_spec_feats)
    elif feature_type == 'harmonic':
        pred_test_base[k_feat] = model_base.predict(test_harmonic_feats)
    elif feature_type == 'percussive':
        pred_test_base[k_feat] = model_base.predict(test_percussive_feats)
    else:
        raise ValueError('Invalid feature type!')

########################################################################################################################
# process and store all results
########################################################################################################################

# leaderboard data
np.save('pred_leaderboard_base.npy', pred_leaderboard_base)

# do closed-set classification
pred_cat_leaderboard_single = np.argmax(pred_leaderboard_base[0], axis=1)
pred_cat_leaderboard_ensemble = np.argmax(gmean(np.concatenate([pred_leaderboard_base, np.expand_dims(pred_leaderboard_base[0], axis=0)], axis=0), axis=0), axis=1)

# estimate outliers
threshold_outlier = 0.5
threshold_closed = 0.5
pred_cat_leaderboard_single[outlier_scores_clf < threshold_outlier] = 10
pred_cat_leaderboard_single[np.max(pred_leaderboard_base[0], axis=1) < threshold_closed] = 10
pred_cat_leaderboard_ensemble[outlier_scores_clf < threshold_outlier] = 10
pred_cat_leaderboard_ensemble[np.max(gmean(np.concatenate([pred_leaderboard_base, np.expand_dims(pred_leaderboard_base[0], axis=0)], axis=0), axis=0), axis=1) < threshold_closed] = 10

# store results
leaderboard_results_single = pd.DataFrame({'Id': np.arange(pred_cat_leaderboard_single.shape[0]),
                                    'Scene_label': categories[pred_cat_leaderboard_single]})
leaderboard_results_single.to_csv('output_leaderboard.csv', encoding='utf-8', index=False)
leaderboard_results_ensemble = pd.DataFrame({'Id': np.arange(pred_cat_leaderboard_ensemble.shape[0]),
                                    'Scene_label': categories[pred_cat_leaderboard_ensemble]})
leaderboard_results_ensemble.to_csv('output_leaderboard_ensemble.csv', encoding='utf-8', index=False)

# test data
np.save('pred_test_base.npy', pred_test_base)

# do closed-set classification
pred_cat_test_single = np.argmax(pred_test_base[0], axis=1)
pred_cat_test_ensemble = np.argmax(gmean(np.concatenate([pred_test_base, np.expand_dims(pred_test_base[0], axis=0)], axis=0), axis=0), axis=1)

# estimate outliers
threshold_outlier = 0.5
threshold_closed = 0.5
pred_cat_test_single[outlier_scores_test_clf[0] < threshold_outlier] = 10
pred_cat_test_single[np.max(pred_test_base[0], axis=1) < threshold_closed] = 10
pred_cat_test_ensemble[gmean(np.concatenate([outlier_scores_test_clf, np.expand_dims(outlier_scores_test_clf[0], axis=0)], axis=0), axis=0) < threshold_outlier] = 10
pred_cat_test_ensemble[np.max(gmean(np.concatenate([pred_test_base, np.expand_dims(pred_test_base[0], axis=0)], axis=0), axis=0), axis=1) < threshold_closed] = 10

# store results
test_results_single = pd.DataFrame({'File': [file_name + "\t" + scene_label for file_name, scene_label in
                                             zip([l.split("\n")[0] for l in open(dataset_test_path + 'evaluation_setup/test.csv','r').readlines()[1:]], categories[pred_cat_test_single])]})
test_results_single.to_csv('output_test_single.csv', encoding='utf-8', index=False, header=False)
test_results_ensemble = pd.DataFrame({'File': [file_name + "\t" + scene_label for file_name, scene_label in
                                               zip([l.split("\n")[0] for l in open(dataset_test_path + 'evaluation_setup/test.csv','r').readlines()[1:]], categories[pred_cat_test_ensemble])]})
test_results_ensemble.to_csv('output_test_ensemble.csv', encoding='utf-8', index=False, header=False)
