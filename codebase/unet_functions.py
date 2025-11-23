import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.path as mpath
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
import calfem.geometry as cfg
import calfem.mesh as cfm
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator, CubicSpline, interp1d, PchipInterpolator, RegularGridInterpolator
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
import sys
import os
import json
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape, Concatenate
from tensorflow.keras.models import Model
from tqdm import tqdm


def unet(input_shape=(32, 32, 2),layer_sizes=(32,64,128)):
    inputs = layers.Input(shape=input_shape)

    # Encoder path
    # Level 1
    c1 = layers.Conv2D(layer_sizes[0], (3,3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2,2))(c1) # 16x16

    # Level 2
    c2 = layers.Conv2D(layer_sizes[1], (3,3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2,2))(c2) # 8x8

    # Level 3 (Bottleneck)
    c3 = layers.Conv2D(layer_sizes[2], (3,3), activation='relu', padding='same')(p2)

    # Decoder path
    # Level 2 up
    u2 = layers.UpSampling2D((2,2))(c3) # 16x16
    u2 = layers.concatenate([u2, c2], axis=-1)
    c4 = layers.Conv2D(layer_sizes[1], (3,3), activation='relu', padding='same')(u2)

    # Level 1 up
    u1 = layers.UpSampling2D((2,2))(c4) # 32x32
    u1 = layers.concatenate([u1, c1], axis=-1)
    c5 = layers.Conv2D(layer_sizes[0], (3,3), activation='relu', padding='same')(u1)

    # Output layer
    # We want output shape (32,32,2), so number of filters = 2
    # Choose activation according to your task (e.g., 'sigmoid', 'softmax', or 'linear')
    outputs = layers.Conv2D(2, (1,1), activation=None)(c5)

    model = Model(inputs, outputs, name='unet_from_scratch')
    return model
