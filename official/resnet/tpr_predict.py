# coding=utf-8
# Copyright (c) 2019 Alo7 Inc. All Rights Reserved.
# ==============================================================================
""" """
import cv2
import tensorflow as tf

export_dir = './tpr_export/1555563916'

img = cv2.imread('./tpr/tpr_9/150/deactive1.jpg')

b, g, r = cv2.split(img)
img = cv2.merge([r, g, b])

resized_image = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
predict_fn = tf.contrib.predictor.from_saved_model(export_dir)
predictions = predict_fn({'input': [resized_image]})

print(predictions)
