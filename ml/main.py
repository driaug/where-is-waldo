from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import sys
import random
import tensorflow as tf
from keras import backend as K
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image


def get_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)


K.set_session(get_session())


def get_waldo_bounds(image):
    model_path = './ml/model/model.h5'
    model = models.load_model(model_path, backbone_name='resnet50')

    image = read_image_bgr(image)
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    image = preprocess_image(image)
    image, scale = resize_image(image, min_side=1800, max_side=3000)

    boxes, scores, labels = model.predict_on_batch(
        np.expand_dims(image, axis=0))

    boxes /= scale

    return boxes
