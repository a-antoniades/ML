import wave
import struct
import os
import numpy as np
import pandas as pd


def pcm_channels(wave_file):
    """Given a file-like object or file path representing a wave file,
    decompose it into its constituent PCM data streams.

    Input: A file like object or file path
    Output: A list of lists of integers representing the PCM coded data stream channels
        and the sample rate of the channels (mixed rate channels not supported)
    """
    stream = wave.open(wave_file, "rb")

    num_channels = stream.getnchannels()
    sample_rate = stream.getframerate()
    sample_width = stream.getsampwidth()
    num_frames = stream.getnframes()

    raw_data = stream.readframes( num_frames ) # Returns byte data
    stream.close()

    total_samples = num_frames * num_channels

    if sample_width == 1:
        fmt = "%iB" % total_samples # read unsigned chars
    elif sample_width == 2:
        fmt = "%ih" % total_samples # read signed 2 byte shorts
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    integer_data = struct.unpack(fmt, raw_data)
    del raw_data # Keep memory tidy (who knows how big it might be)

    channels = [[] for time in range(num_channels)]

    for index, value in enumerate(integer_data):
        bucket = index % num_channels
        channels[bucket].append(value)

    return channels


def sound_array(folder):
    soundArray = []
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):
            sound = pcm_channels(os.path.join(folder, filename))
            soundArray.append(sound[0])

    return soundArray


soundArray0 = sound_array("/Users/antonis/Desktop/Themas/Sound/sound samples")
soundArray0 = np.array(soundArray0)
soundArray0df = numpy_fillnna(soundArray0)





rec1 = pcm_channels("/Users/antonis/Desktop/Themas/Sound/DP1.wav")
rec2 = pcm_channels("/Users/antonis/Desktop/Themas/Sound/DP22.wav")
rec3 = pcm_channels("/Users/antonis/Desktop/Themas/Sound/DP2.wav")



# Data manipulation

recz = rec1[0]
rec = rec1[0], rec2[0], rec3[0]
arraydf = np.array(rec)


def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out


arraydf = numpy_fillna(arraydf)

# rec = np.vstack((rec11, rec22, rec33))


rec1Xtrain = pd.DataFrame(arraydf)
rec11Ytrain = "rec1"


model = XGB(Classifier)
model.fit