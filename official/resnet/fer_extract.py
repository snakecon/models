import csv
import math
import os
import sys
from itertools import islice
from os import path

import numpy as np
import tensorflow as tf
from PIL import Image

TRAINING_SHARDS = 60
VALIDATION_SHARDS = 30

TRAIN_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'validation'
# List of folders for training, validation and test.
folder_names = {'Training': 'FER2013Train',
                'PublicTest': 'FER2013Valid',
                'PrivateTest': 'FER2013Test'}


def _process_data(emotion_raw, mode):
    '''
    Based on https://arxiv.org/abs/1608.01041, we process the data differently depend on the training mode:

    Majority: return the emotion that has the majority vote, or unknown if the count is too little.
    Probability or Crossentropty: convert the count into probability distribution.abs
    Multi-target: treat all emotion with 30% or more votes as equal.
    '''
    size = len(emotion_raw)
    emotion_unknown = [0.0] * size
    emotion_unknown[-2] = 1.0

    # remove emotions with a single vote (outlier removal)
    for i in range(size):
        if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
            emotion_raw[i] = 0.0

    sum_list = sum(emotion_raw)
    emotion = [0.0] * size

    if mode == 'majority':
        # find the peak value of the emo_raw list
        maxval = max(emotion_raw)
        if maxval > 0.5 * sum_list:
            emotion[np.argmax(emotion_raw)] = maxval
        else:
            emotion = emotion_unknown  # force setting as unknown
    elif (mode == 'probability') or (mode == 'crossentropy'):
        sum_part = 0
        count = 0
        valid_emotion = True
        while sum_part < 0.75 * sum_list and count < 3 and valid_emotion:
            maxval = max(emotion_raw)
            for i in range(size):
                if emotion_raw[i] == maxval:
                    emotion[i] = maxval
                    emotion_raw[i] = 0
                    sum_part += emotion[i]
                    count += 1
                    if i >= 8:  # unknown or non-face share same number of max votes
                        valid_emotion = False
                        if sum(
                                emotion) > maxval:  # there have been other emotions ahead of unknown or non-face
                            emotion[i] = 0
                            count -= 1
                        break
        if sum(
                emotion) <= 0.5 * sum_list or count > 3:  # less than 50% of the votes are integrated, or there are too many emotions, we'd better discard this example
            emotion = emotion_unknown  # force setting as unknown
    elif mode == 'multi_target':
        threshold = 0.3
        for i in range(size):
            if emotion_raw[i] >= threshold * sum_list:
                emotion[i] = emotion_raw[i]
        if sum(
                emotion) <= 0.5 * sum_list:  # less than 50% of the votes are integrated, we discard this example
            emotion = emotion_unknown  # set as unknown

    return [float(i) / sum(emotion) for i in emotion]


def _check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, synset, height, width):
    """Build an Example proto for an example.

    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """
    colorspace = b'RGB'
    channels = 3
    image_format = b'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/synset': _bytes_feature(bytes(synset, 'ascii')),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(
            bytes(os.path.basename(filename), 'ascii')),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example


def _is_png(filename):
    """Determine if a file contains a PNG format image.

    Args:
      filename: string, path of the image file.

    Returns:
      boolean indicating if the image is a PNG.
    """
    # File list from:
    # https://github.com/cytsai/ilsvrc-cmyk-image-list
    return filename.endswith('png')


def _process_image(filename, coder):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Crop image's black boarder.
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Clean the dirty data.
    if _is_png(filename):
        # 1 image is a PNG.
        tf.logging.info('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def _process_image_files_batch(coder, output_file, filenames, synsets, labels):
    """Processes and saves list of images as TFRecords.

    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      output_file: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      synsets: list of strings; each string is a unique WordNet ID
      labels: map of string to integer; id for all synset labels
    """
    writer = tf.python_io.TFRecordWriter(output_file)

    for filename, synset in zip(filenames, synsets):
        image_buffer, height, width = _process_image(filename, coder)

        label_list = _process_data(list(int(x) for x in synset.split(',')),
                                   'majority')
        label = np.argmax(label_list) + 1

        if label > len(labels):
            # Skip unknown(9) or no-face(10).
            continue

        # label = labels[synset]
        example = _convert_to_example(filename, image_buffer, label,
                                      synset, height, width)
        writer.write(example.SerializeToString())

    writer.close()


def _process_dataset(filenames, synsets, labels, output_directory, prefix,
                     num_shards):
    """Processes and saves list of images as TFRecords.

    Args:
      filenames: list of strings; each string is a path to an image file
      synsets: list of strings; each string is a unique WordNet ID
      labels: map of string to integer; id for all synset labels
      output_directory: path where output files should be created
      prefix: string; prefix for each file
      num_shards: number of chucks to split the filenames into

    Returns:
      files: list of tf-record filepaths created from processing the dataset.
    """
    _check_or_create_dir(output_directory)
    chunksize = int(math.ceil(len(filenames) / num_shards))
    coder = ImageCoder()

    files = []

    for shard in range(num_shards):
        chunk_files = filenames[shard * chunksize: (shard + 1) * chunksize]
        chunk_synsets = synsets[shard * chunksize: (shard + 1) * chunksize]
        output_file = os.path.join(
            output_directory, '%s-%.5d-of-%.5d' % (prefix, shard, num_shards))
        _process_image_files_batch(coder, output_file, chunk_files,
                                   chunk_synsets, labels)
        tf.logging.info('Finished writing file: %s' % output_file)
        files.append(output_file)
    return files


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb',
                                                 quality=100)

        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb',
                                                 quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data,
                                                 channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._sess.run(self._cmyk_to_rgb,
                              feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


class DataConverter(object):
    def __init__(self):
        pass

    def str_to_image(self, image_blob):
        ''' Convert a string blob to an image object. '''
        image_string = image_blob.split(' ')
        image_data = np.asarray(image_string, dtype=np.uint8).reshape(48, 48)
        return Image.fromarray(image_data)

    def preprocess_fer(self, base_folder, ferplus_path, fer_path):
        print("Start generating ferplus images.")

        for key, value in folder_names.items():
            folder_path = os.path.join(base_folder, value)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        ferplus_entries = []
        with open(ferplus_path, 'r') as csvfile:
            ferplus_rows = csv.reader(csvfile, delimiter=',')
            for row in islice(ferplus_rows, 1, None):
                ferplus_entries.append(row)

        index = 0
        with open(fer_path, 'r') as csvfile:
            fer_rows = csv.reader(csvfile, delimiter=',')
            for row in islice(fer_rows, 1, None):
                ferplus_row = ferplus_entries[index]
                file_name = ferplus_row[1].strip()
                if len(file_name) > 0:
                    image = self.str_to_image(row[1])
                    image_path = os.path.join(base_folder, folder_names[row[2]],
                                              file_name)
                    image.save(image_path, compress_level=0)
                index += 1

        print("Done...")

    def to_tf_records(self, raw_data_dir, local_scratch_dir,
                      train_names, validation_names, ferplus_path):
        ferplus_entries = {}
        with open(ferplus_path, 'r') as csvfile:
            ferplus_rows = csv.reader(csvfile, delimiter=',')
            for row in islice(ferplus_rows, 1, None):
                k = row[1]
                v = ','.join(row[2:])
                ferplus_entries[k] = v

        # Analyze pics.
        train_files = []
        validation_files = []
        train_synsets = []
        validation_synsets = []

        for root, dirs, files in os.walk(raw_data_dir):
            if len(dirs) != 0:
                continue
            root_parts = root.split('/')
            assert len(root_parts) == 2
            bucket_name = root_parts[1]
            if bucket_name in train_names:
                for file_name in files:
                    if file_name.endswith('png'):
                        train_files.append(path.join(root, file_name))
                        train_synsets.append(ferplus_entries[file_name])
            if bucket_name in validation_names:
                for file_name in files:
                    if file_name.endswith('png'):
                        validation_files.append(path.join(root, file_name))
                        validation_synsets.append(ferplus_entries[file_name])

        # Create unique ids for all synsets
        labels = {
            'neutral': 0,
            'happiness': 1,
            'surprise': 2,
            'sadness': 3,
            'anger': 4,
            'disgust': 5,
            'fear': 6,
            'contempt': 7
        }

        # Create tf_record data
        train_records = _process_dataset(
            train_files, train_synsets, labels,
            os.path.join(local_scratch_dir, TRAIN_DIRECTORY),
            TRAIN_DIRECTORY, TRAINING_SHARDS)
        validation_records = _process_dataset(
            validation_files, validation_synsets, labels,
            os.path.join(local_scratch_dir, VALIDATION_DIRECTORY),
            VALIDATION_DIRECTORY, VALIDATION_SHARDS)

        return train_records, validation_records


if __name__ == '__main__':
    data_converter = DataConverter()

    # data_converter.preprocess_fer('fer', 'fer/fer2013new.csv',
    #                               'fer/fer2013/fer2013.csv')

    train_records, validation_records = data_converter.to_tf_records(
        'fer', 'fer_dataset', {'FER2013Train'}, {'FER2013Valid'},
        'fer/fer2013new.csv'
    )
    print(train_records)
    print(validation_records)
