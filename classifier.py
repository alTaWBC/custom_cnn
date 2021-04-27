from os import listdir
from os.path import isfile, join
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


QUARTER_SECOND = 5120
SAMPLE_RATE = 44100

isFileLocation = True
BEST_MODEL = r"C:\Users\WilliamCosta\Desktop\repositories\updated_tensorflow\updated_model\vocal_folds"


def splitSignal(signal):
    signals = tf.reshape(signal, [1, -1])
    # add two seconds of padding, only zeros
    signals = tf.pad(
        signals, [[0, 0], [0, QUARTER_SECOND]], "CONSTANT", constant_values=0)

    # cut the audio signal to two seconds
    return signals[:, :QUARTER_SECOND]


def classification_to_label(place, voice):
    labels = [['S', 'J'], ['Z', 'CH']]
    return labels[np.argmax(voice)][np.argmax(place)]


def shortTimeFourierTransformationOfSignals(signals):
    # `stfts` is a complex64 Tensor representing the Short-time Fourier Transform of
    # each signal in `signals`. Its shape is [batch_size, ?, fft_unique_bins]
    # where fft_unique_bins = fft_length // 2 + 1 = 513.
    return tf.signal.stft(
        signals, frame_length=1024, frame_step=512, fft_length=1024)


def getSpectrogramAndBins(stfts):
    # An energy spectrogram is the magnitude of the complex-valued STFT.
    # A float32 Tensor of shape [batch_size, ?, 513].
    magnitude_spectrograms = tf.abs(stfts)

    # Warp the linear-scale, magnitude spectrograms into the mel-scale.
    return magnitude_spectrograms, magnitude_spectrograms.shape[-1].value


def getMelSpectograms(magnitude_spectrograms, num_spectrogram_bins):
    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, SAMPLE_RATE,
        lower_edge_hertz, upper_edge_hertz)

    mel_spectrograms = tf.tensordot(
        magnitude_spectrograms, linear_to_mel_weight_matrix, 1)

    mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    log_offset = 1e-6
    log_mel_spectrogram = tf.math.log(mel_spectrograms + log_offset)
    return log_mel_spectrogram


def audio_signal_to_log_mel(audio_signal):

    if isFileLocation:
        audio_signal = tf.compat.v1.read_file(audio_signal)

    wav_decoder = tf.audio.decode_wav(audio_signal, desired_channels=1)
    signals = splitSignal(wav_decoder.audio)
    print(wav_decoder.sample_rate)
    stfts = shortTimeFourierTransformationOfSignals(signals)
    print(getSpectrogramAndBins(stfts))
    return getMelSpectograms(*getSpectrogramAndBins(stfts))


# %%% Init

tf.compat.v1.disable_v2_behavior()
session = tf.compat.v1.Session()

best_model = load_model(BEST_MODEL)

best_model._make_predict_function()

audio_signal = tf.compat.v1.placeholder(tf.string)
log_mel_spectrogram = audio_signal_to_log_mel(audio_signal)


def generateLogMel(audio_bytes):
    log_mel = session.run(log_mel_spectrogram, feed_dict={
        audio_signal: audio_bytes})
    log_mel = log_mel.reshape(
        log_mel.shape[0], log_mel.shape[1], log_mel.shape[2], 1)
    return np.swapaxes(log_mel, 1, 2)


def classify(log_mel):
    probabilities = best_model.predict(log_mel)
    return ['s', 'z', 'ch', 'j'][min(3, np.argmax(probabilities))]


def classifySound(audio_bytes):
    return classify(generateLogMel(audio_bytes))


if __name__ == '__main__':
    # best_model.summary()
    # # print(classify(np.zeros((1, 80, 9, 1))))
    # # file_loc = r"C:\Users\WilliamCosta\Desktop\repositories\tese\assets\sounds\data\raw\001_CH_L_R.wav"
    # # print(classifySound(file_loc))
    # for layer in best_model.layers:
    #     print(layer.name)
    file_loc = r"C:\Users\WilliamCosta\Desktop\repositories\tese\assets\sounds\data\raw\001_CH_L_R.wav"
    np.save('log_mel', generateLogMel(file_loc))
    print(best_model.predict(generateLogMel(file_loc)))
