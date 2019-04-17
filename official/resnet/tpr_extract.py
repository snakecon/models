import math
import os
import re
from os import path

import cv2
import numpy as np
import tensorflow as tf
from scipy.stats import mode

TRAINING_SHARDS = 60
VALIDATION_SHARDS = 30

TRAIN_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'validation'


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


def crop(origin_path, save_path):
    img = cv2.imread(origin_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    closed_1 = cv2.erode(gray, None, iterations=4)
    closed_1 = cv2.dilate(closed_1, None, iterations=4)
    blurred = cv2.blur(closed_1, (9, 9))
    # get the most frequent pixel
    num = mode(blurred.flat)[0][0] + 1
    # the threshold depends on the mode of your images' pixels
    num = num if num <= 30 else 1

    _, thresh = cv2.threshold(blurred, num, 255, cv2.THRESH_BINARY)

    # you can control the size of kernel according your need.
    kernel = np.ones((13, 13), np.uint8)
    closed_2 = cv2.erode(thresh, kernel, iterations=4)
    closed_2 = cv2.dilate(closed_2, kernel, iterations=4)

    _, cnts, _ = cv2.findContours(closed_2.copy(), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    # draw a bounding box arounded the detected barcode and display the image
    # cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
    # cv2.imshow("Image", img)
    # cv2.imwrite("pic.jpg", img)
    # cv2.waitKey(0)

    xs = [i[0] for i in box]
    ys = [i[1] for i in box]
    x1 = min(xs)
    x2 = max(xs)
    y1 = min(ys)
    y2 = max(ys)
    height = y2 - y1
    width = x2 - x1
    crop_img = img[y1:y1 + height, x1:x1 + width]
    cv2.imwrite(save_path, crop_img)
    print('Crop %s to %s' % (origin_path, save_path))


def _process_image(src_filename, coder):
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
    filename = src_filename.replace('png', 'jpg')
    crop(src_filename, filename)

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
        label = labels[synset]
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

    def to_tf_records(self, raw_data_dir, local_scratch_dir,
                      train_names, validation_names):
        # Analyze pics.
        train_files = []
        validation_files = []
        train_synsets = []
        validation_synsets = []

        for root, dirs, files in os.walk(raw_data_dir):
            if len(dirs) != 0:
                continue
            root_parts = root.split('/')
            assert len(root_parts) == 3
            bucket_name = root_parts[1]
            if bucket_name in train_names:
                for file_name in files:
                    if file_name.endswith('png'):
                        train_files.append(path.join(root, file_name))
                        train_synsets.append(re.findall(r"(.+?)\d+.png", file_name)[0])
            if bucket_name in validation_names:
                for file_name in files:
                    if file_name.endswith('png'):
                        validation_files.append(path.join(root, file_name))
                        validation_synsets.append(re.findall(r"(.+?)\d+.png", file_name)[0])

        # Check label status
        for file_name, synetts in zip(train_files, train_synsets):
            if synetts not in {'active', 'deactive'}:
                print(file_name)

        for file_name, synetts in zip(validation_files, validation_synsets):
            if synetts not in {'active', 'deactive'}:
                print(file_name)

        # Create unique ids for all synsets
        labels = {v: k + 1 for k, v in enumerate(
            sorted(set(train_synsets + validation_synsets)))}

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
    train_records, validation_records = data_converter.to_tf_records(
        'tpr', 'tpr_dataset', {'tpr_1', 'tpr_2', 'tpr_5'}, {'tpr_9'}
    )
    print(train_records)
    print(validation_records)