import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import StratifiedKFold
import keras
from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser
import matplotlib.pyplot as plt
import os
from keras.engine.topology import Layer
from keras import backend as K
import tensorflow as tf
from scipy.signal import medfilt2d


class LogMelExtractor(object):
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


class FusionLayer(Layer):
    """
    Custom Layer for fusing scores via Logistic Regression in a neural network.
    """
    def __init__(self, num_classes, num_models, **kwargs):
        self.num_classes = num_classes
        self.num_models = num_models
        super(FusionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.fusion_weights = self.add_weight(name='fusion_weights', shape=(self.num_models+1,), initializer='uniform',
                                              trainable=True, regularizer=keras.regularizers.l2(0.0005))
        super(FusionLayer, self).build(input_shape)

    def call(self, x):
        output = x[:, 0:self.num_classes]*self.fusion_weights[1]+self.fusion_weights[0]
        for k in np.arange(1, self.num_models):
            output = output + x[:, k*self.num_classes:self.num_classes*(k+1)]*self.fusion_weights[k+1]
        return K.softmax(output)

    def get_config(self):
        config = {'num_classes': self.num_classes, 'num_models': self.num_models}
        base_config = super(FusionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_classes)


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
    #x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv2D(64, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 3))(x)
    #x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv2D(128, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    #x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv2D(128, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 3))(x)
    #x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv2D(196, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    #x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv2D(196, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 3))(x)
    #x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv2D(256, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    #x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv2D(256, kernel_size=3, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    #x = keras.layers.AveragePooling2D(pool_size=(2, 3))(x)
    #x = keras.layers.Dropout(0.3)(x)
    #x = keras.layers.Flatten()(x)

    #x = keras.layers.Dense(100)(x)
    #x = keras.layers.BatchNormalization()(x)
    #x = keras.layers.Activation(activation='relu')(x)
    #x = keras.layers.Dropout(0.3)(x)
    model_output = keras.layers.Dense(10, activation='softmax')(x)
    """
    x = keras.layers.Conv2D(32, kernel_size=7, padding='same')(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=5)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv2D(64, kernel_size=7, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(4, 100))(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(100)(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    model_output = keras.layers.Dense(10, activation='softmax', kernel_initializer='uniform')(x)
    """
    return input, model_output



########################################################################################################################
# Load data and compute spectrograms
########################################################################################################################

nceps_mel_spec = 64
background_subtraction_kernel_size = (43, 11)
# load train data
print('Loading train data')
dataset_path = 'D:/dcase2019_task1_baseline-master/datasets/TAU-urban-acoustic-scenes-2019-development/'
train_sheet = pd.DataFrame({'File': [l.split("\t")[0].split("\n")[0] for l in open(dataset_path + 'evaluation_setup/fold1_train.csv','r').readlines()[1:]],
                          'Category': [l.split("\t")[1].split("\n")[0] for l in open(dataset_path + 'evaluation_setup/fold1_train.csv','r').readlines()[1:]]})
categories = np.unique(train_sheet.Category)
#print(categories)
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

# shuffle data randomly
train_labels = np.concatenate([train_labels, eval_labels], axis=0)
train_mel_spec_feats = np.concatenate([train_mel_spec_feats, eval_mel_spec_feats], axis=0)
train_harmonic_feats = np.concatenate([train_harmonic_feats, eval_harmonic_feats], axis=0)
train_percussive_feats = np.concatenate([train_percussive_feats, eval_percussive_feats], axis=0)
shuffled_index = np.random.permutation(len(train_labels))
train_labels = train_labels[shuffled_index]
train_mel_spec_feats = train_mel_spec_feats[shuffled_index]
train_harmonic_feats = train_harmonic_feats[shuffled_index]
train_percussive_feats = train_percussive_feats[shuffled_index]



########################################################################################################################
# Preprocessing
########################################################################################################################
# feature normalization
print('Normalizing data')
eps = 1e-12
mean_mel_spec = np.mean(train_mel_spec_feats, axis=0)
std_mel_spec = np.std(train_mel_spec_feats, axis=0)
for k in np.arange(train_mel_spec_feats.shape[0]):
    train_mel_spec_feats[k] = (train_mel_spec_feats[k]-mean_mel_spec)/(std_mel_spec+eps)
for k in np.arange(eval_mel_spec_feats.shape[0]):
    eval_mel_spec_feats[k] = (eval_mel_spec_feats[k]-mean_mel_spec)/(std_mel_spec+eps)
mean_harmonic = np.mean(train_harmonic_feats, axis=0)
std_harmonic = np.std(train_harmonic_feats, axis=0)
for k in np.arange(train_harmonic_feats.shape[0]):
    train_harmonic_feats[k] = (train_harmonic_feats[k]-mean_harmonic)/(std_harmonic+eps)
for k in np.arange(eval_harmonic_feats.shape[0]):
    eval_harmonic_feats[k] = (eval_harmonic_feats[k]-mean_harmonic)/(std_harmonic+eps)
mean_percussive = np.mean(train_percussive_feats, axis=0)
std_percussive = np.std(train_percussive_feats, axis=0)
for k in np.arange(train_percussive_feats.shape[0]):
    train_percussive_feats[k] = (train_percussive_feats[k]-mean_percussive)/(std_percussive+eps)
for k in np.arange(eval_percussive_feats.shape[0]):
    eval_percussive_feats[k] = (eval_percussive_feats[k]-mean_percussive)/(std_percussive+eps)

"""
# background subtraction
print('Applying background subtraction to train data')
for idx in np.arange(train_mel_spec_feats.shape[0]):
    train_mel_spec_feats[idx, :, :, 0] = train_mel_spec_feats[idx, :, :, 0] -\
                                         medfilt2d(train_mel_spec_feats[idx, :, :, 0],
                                                   kernel_size=background_subtraction_kernel_size)
print('Applying background subtraction to evaluation data')
for idx in np.arange(eval_mel_spec_feats.shape[0]):
    eval_mel_spec_feats[idx, :, :, 0] = eval_mel_spec_feats[idx, :, :, 0] -\
                                        medfilt2d(eval_mel_spec_feats[idx, :, :, 0],
                                                  kernel_size=background_subtraction_kernel_size)
"""

########################################################################################################################
# train base cnn on all data
########################################################################################################################
batch_size = 16
batch_size_test = 16
epochs = 1000
aeons = 10
alpha = 1
y_train_cat = keras.utils.np_utils.to_categorical(train_labels, num_classes=len(categories))
y_eval_cat = keras.utils.np_utils.to_categorical(eval_labels, num_classes=len(categories))
# compile model
input, model_output = model_cnn(442, nceps=nceps_mel_spec)
model_base = keras.Model(inputs=[input], outputs=[model_output])
model_base.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(decay=0.0001),
                       metrics=['accuracy'])
print(model_base.summary())

# feature_type = 'mel_spec'
feature_type = 'harmonic'
# feature_type = 'percussive'
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
# create data generator for mixup and random erasing for every batch
datagen_mel_spec = keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.6,  # up to 60 percent shift in time
    height_shift_range=3,  # up to 3 mel bins up or down
    preprocessing_function=get_random_eraser(v_l=np.min(train_data), v_h=np.max(train_data))
    )
test_datagen_mel_spec = keras.preprocessing.image.ImageDataGenerator(
)
datagen_mel_spec.fit(np.r_[train_data])
test_datagen_mel_spec.mean, test_datagen_mel_spec.std = datagen_mel_spec.mean, datagen_mel_spec.std
training_generator_mel_spec = MixupGenerator(train_data, y_train_cat, batch_size=batch_size,
                                             alpha=alpha, datagen=datagen_mel_spec)

# checkpoint = keras.callbacks.ModelCheckpoint('wts_mel_spec.h5', monitor='val_loss', save_best_only=True,)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs_base', histogram_freq=0,
                                                   write_graph=True,
                                                   write_images=False)

for k in np.arange(aeons):
    # fit model
    weight_path = 'wts_' + feature_type + '_' + str(k+1) + 'k.h5'
    if not os.path.isfile(weight_path):
        model_base.fit_generator(generator=training_generator_mel_spec(), verbose=2,
                                          callbacks=[tensorboard_callback#, checkpoint
                            ],
                                          steps_per_epoch=train_data.shape[0] // batch_size, epochs=epochs,
                                          #validation_data=test_datagen_mel_spec.flow(eval_data, y_eval_cat)
                          )
        model_base.save(weight_path)
    else:
        model_base = keras.models.load_model(weight_path)

    # evaluate model
    pred_base = model_base.predict_generator(
        generator=test_datagen_mel_spec.flow(eval_data, y_eval_cat, shuffle=False))
    acc = np.mean(np.argmax(pred_base, axis=1) == np.argmax(y_eval_cat, axis=1))
    print('Accuracy of base model is ' + str(acc))

########################################################################################################################
# predict on leaderboard data
########################################################################################################################
# load leaderboard data
print('Loading leaderboard data')
dataset_leaderboard_path = 'D:/dcase2019_task1_baseline-master/datasets/TAU-urban-acoustic-scenes-2019-leaderboard/'
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

# normalize data
print('Normalizing data')
for k in np.arange(leaderboard_mel_spec_feats.shape[0]):
    leaderboard_mel_spec_feats[k] = (leaderboard_mel_spec_feats[k]-mean_mel_spec)/(std_mel_spec+eps)
for k in np.arange(leaderboard_harmonic_feats.shape[0]):
    leaderboard_harmonic_feats[k] = (leaderboard_harmonic_feats[k]-mean_harmonic)/(std_harmonic+eps)
for k in np.arange(leaderboard_percussive_feats.shape[0]):
    leaderboard_percussive_feats[k] = (leaderboard_percussive_feats[k]-mean_percussive)/(std_percussive+eps)

# predict
print('Predicting on leaderboard data')
if feature_type == 'mel_spec':
    pred_leaderboard_base = model_base.predict(leaderboard_mel_spec_feats)
elif feature_type == 'harmonic':
    pred_leaderboard_base = model_base.predict(leaderboard_harmonic_feats)
elif feature_type == 'percussive':
    pred_leaderboard_base = model_base.predict(leaderboard_percussive_feats)
else:
    raise ValueError('Invalid feature type!')
pred_cat_leaderboard = np.argmax(pred_leaderboard_base, axis=1)

# store results in .csv file
leaderboard_results = pd.DataFrame({'Id': np.arange(pred_cat_leaderboard.shape[0]),
                                    'Scene_label': categories[pred_cat_leaderboard]})
leaderboard_results.to_csv('output_leaderboard.csv', encoding='utf-8', index=False)
