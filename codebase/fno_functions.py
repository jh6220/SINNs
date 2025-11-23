import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.path as mpath
import plotly.express as px
import plotly.graph_objects as go
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
from tensorflow import keras
from tqdm import tqdm


class SpectralConv1D(layers.Layer):
    """
    1D spectral convolution used in FNO.
    - FFT along the length axis (not channels).
    - Only the lowest `modes` frequencies are mixed with learned complex weights.
    - Channel mixing per mode.
    Output has the same channels as input (width is fixed across the block stack).
    """
    def __init__(self, modes: int, **kwargs):
        super().__init__(**kwargs)
        self.modes = int(modes)

    def build(self, input_shape):
        # input: (B, L, C)
        _, L, C = input_shape
        self.C = int(C)
        self.modes = min(self.modes, L // 2)  # safety

        # Per-mode complex weights for positive and negative frequencies
        wshape = (self.modes, self.C, self.C)
        init = keras.initializers.GlorotUniform()
        self.w_pos_real = self.add_weight("w_pos_real", shape=wshape, initializer=init)
        self.w_pos_imag = self.add_weight("w_pos_imag", shape=wshape, initializer=init)
        self.w_neg_real = self.add_weight("w_neg_real", shape=wshape, initializer=init)
        self.w_neg_imag = self.add_weight("w_neg_imag", shape=wshape, initializer=init)

        # pointwise (physical space) mixing to complement low-mode spectral mixing
        self.w_point = layers.Dense(self.C, use_bias=True)

    def call(self, x):
        # x: (B, L, C)
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]

        # FFT along length (we move length to last axis for tf.signal.fft)
        x_c = tf.cast(x, tf.complex64)               # (B,L,C)
        x_c = tf.transpose(x_c, [0, 2, 1])           # (B,C,L)
        x_ft = tf.signal.fft(x_c)                    # (B,C,L)

        # Prepare outputs in frequency domain
        out_ft = tf.zeros_like(x_ft)

        if self.modes > 0:
            # Positive low modes ([:modes])
            w_pos = tf.complex(self.w_pos_real, self.w_pos_imag)  # (m,C,C)
            pos = tf.transpose(x_ft[..., :self.modes], [0, 2, 1]) # (B,m,C)
            out_pos = tf.einsum('bmc,mco->bmo', pos, w_pos)       # (B,m,C)
            out_pos = tf.transpose(out_pos, [0, 2, 1])            # (B,C,m)

            # Negative low modes ([-modes:])
            w_neg = tf.complex(self.w_neg_real, self.w_neg_imag)  # (m,C,C)
            neg = tf.transpose(x_ft[..., -self.modes:], [0, 2, 1])# (B,m,C)
            out_neg = tf.einsum('bmc,mco->bmo', neg, w_neg)       # (B,m,C)
            out_neg = tf.transpose(out_neg, [0, 2, 1])            # (B,C,m)

            middle = tf.shape(x_ft)[-1] - 2 * self.modes
            out_ft = tf.concat(
                [out_pos, tf.zeros([B, self.C, middle], tf.complex64), out_neg], axis=-1
            )
        # iFFT back to physical space
        y = tf.signal.ifft(out_ft)                   # (B,C,L)
        y = tf.transpose(tf.math.real(y), [0, 2, 1]) # (B,L,C)

        # Add pointwise (physical) mixing
        y = y + self.w_point(x)
        return y


class FNO1DBlock(layers.Layer):
    """FNO-1D block: spectral mixing + residual + GELU."""
    def __init__(self, modes: int, **kwargs):
        super().__init__(**kwargs)
        self.sconv = SpectralConv1D(modes)
        self.norm = layers.LayerNormalization(axis=-1)

    def call(self, x):
        y = self.sconv(x)
        y = self.norm(y)
        y = tf.nn.gelu(y)
        return x + y


class SideEncoder1D(layers.Layer):
    """
    Side-specific 1D encoder:
    Dense projection to width, then L FNO1D blocks.
    """
    def __init__(self, width: int, depth: int, modes: int, **kwargs):
        super().__init__(**kwargs)
        self.width = int(width)
        self.depth = int(depth)
        self.modes = int(modes)
        self.proj = layers.Dense(self.width, use_bias=True)
        self.blocks = [FNO1DBlock(self.modes) for _ in range(self.depth)]
        self.out_norm = layers.LayerNormalization(axis=-1)

    def call(self, x):
        # x: (B, L, Cin)
        y = self.proj(x)  # (B, L, width)
        for blk in self.blocks:
            y = blk(y)
        return self.out_norm(y)  # (B, L, width)


class AxisProfile(layers.Layer):
    """
    Learned 1D profile over an axis (coordinates in [-1,1]) that expands
    a side latent into the interior.
    Produces a matrix of shape (Length, width).
    """
    def __init__(self, length: int, width: int, hidden: int = 32, name=None):
        super().__init__(name=name)
        self.length = int(length)
        self.width = int(width)
        self.hidden = int(hidden)
        # Tiny MLP mapping coords -> width-dim profile at each location
        self.d1 = layers.Dense(self.hidden, activation='gelu')
        self.d2 = layers.Dense(self.width, activation=None)

        # Precompute coordinates (constant)
        y = tf.linspace(-1.0, 1.0, self.length)
        self.coords = tf.reshape(y, [self.length, 1])  # (L,1)

    def call(self, _):
        # Returns (Length, width)
        h = self.d1(self.coords)
        return self.d2(h)


class LiftHorizontal(layers.Layer):
    """Lift a horizontal side feature (N, W, E) to (N, H, W, E) using a vertical profile φ(y,E)."""
    def __init__(self, H: int, E: int, hidden: int = 32, name=None):
        super().__init__(name=name)
        self.H = int(H)
        self.E = int(E)
        self.profile = AxisProfile(self.H, self.E, hidden=hidden, name=f"{name}_profile")

    def call(self, f_x):
        # f_x: (N, W, E), φ: (H, E)
        phi = self.profile(None)                       # (H, E)
        return tf.einsum('nxe,ye->nyxe', f_x, phi)     # (N,H,W,E)


class LiftVertical(layers.Layer):
    """Lift a vertical side feature (N, H, E) to (N, H, W, E) using a horizontal profile ψ(x,E)."""
    def __init__(self, W: int, E: int, hidden: int = 32, name=None):
        super().__init__(name=name)
        self.W = int(W)
        self.E = int(E)
        self.profile = AxisProfile(self.W, self.E, hidden=hidden, name=f"{name}_profile")

    def call(self, f_y):
        # f_y: (N, H, E), ψ: (W, E)
        psi = self.profile(None)                       # (W, E)
        return tf.einsum('nye,xe->nyxe', f_y, psi)     # (N,H,W,E)


class CrossTerm(layers.Layer):
    """
    Build separable cross term w_cross = sum_k v_x^{(k)}(x) ⊗ v_y^{(k)}(y).
    Here we produce a single width = cross_width (i.e., rank = cross_width with 1 channel each),
    implemented as channel-wise outer product.
    """
    def __init__(self, cross_width: int, **kwargs):
        super().__init__(**kwargs)
        self.cross_width = int(cross_width)
        self.fuse_x = layers.Dense(self.cross_width, activation='gelu')
        self.fuse_y = layers.Dense(self.cross_width, activation='gelu')

    def call(self, f_top, f_bot, f_left, f_right):
        # f_top, f_bot: (N, W, E)     -> fuse along channels to v_x: (N, W, Cx)
        # f_left, f_right: (N, H, E)  -> fuse along channels to v_y: (N, H, Cy)
        vx = self.fuse_x(tf.concat([f_top, f_bot], axis=-1))    # (N, W, Cx)
        vy = self.fuse_y(tf.concat([f_left, f_right], axis=-1)) # (N, H, Cy)
        # outer product along x and y for each channel
        w_cross = tf.einsum('nxe,nye->nyxe', vx, vy)            # (N, H, W, Cx==Cy==cross_width)
        return w_cross


class AddCoords2D(layers.Layer):
    """Append normalized (x,y) coordinate channels to a (N,H,W,C) tensor."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        _, H, W, _ = input_shape
        # grids in [-1,1]
        xs = tf.linspace(-1.0, 1.0, W)
        ys = tf.linspace(-1.0, 1.0, H)
        X, Y = tf.meshgrid(xs, ys)           # (H,W)
        self.xy = tf.stack([X, Y], axis=-1)  # (H,W,2)

    def call(self, x):
        B = tf.shape(x)[0]
        xy = tf.tile(self.xy[None, ...], [B, 1, 1, 1])  # (B,H,W,2)
        return tf.concat([x, tf.cast(xy, x.dtype)], axis=-1)


class SpectralConv2D(layers.Layer):
    """
    2D spectral convolution used in FNO.
    - Full complex FFT2D on (H,W).
    - Mix the lowest (modes_h, modes_w) frequencies in the four low-frequency corners.
    - Channel mixing per frequency using complex weights.
    """
    def __init__(self, modes_h: int, modes_w: int, **kwargs):
        super().__init__(**kwargs)
        self.modes_h = int(modes_h)
        self.modes_w = int(modes_w)

    def build(self, input_shape):
        # input: (B, H, W, C)
        _, H, W, C = input_shape
        self.C = int(C)
        self.modes_h = min(self.modes_h, H // 2)
        self.modes_w = min(self.modes_w, W // 2)

        wshape = (self.modes_h, self.modes_w, self.C, self.C)
        init = keras.initializers.GlorotUniform()

        # Four quadrants (top-left, top-right, bottom-left, bottom-right)
        def add_complex(name):
            wr = self.add_weight(f"{name}_real", shape=wshape, initializer=init)
            wi = self.add_weight(f"{name}_imag", shape=wshape, initializer=init)
            return wr, wi

        self.w00_r, self.w00_i = add_complex("w00")
        self.w01_r, self.w01_i = add_complex("w01")
        self.w10_r, self.w10_i = add_complex("w10")
        self.w11_r, self.w11_i = add_complex("w11")

        self.w_point = layers.Conv2D(self.C, kernel_size=1, padding="same", use_bias=True)

    def call(self, x):
        # x: (B,H,W,C)
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]

        # Move channels ahead of spatial for easier einsum
        xc = tf.transpose(tf.cast(x, tf.complex64), [0, 3, 1, 2])      # (B,C,H,W)
        xft = tf.signal.fft2d(xc)                                      # (B,C,H,W)

        mh, mw = self.modes_h, self.modes_w
        zeros = tf.zeros_like(xft)

        # Helper to place a small (B,C,mh,mw) patch into the full (B,C,H,W) tensor
        def place(patch, top, left):
            pad_top = top
            pad_bottom = H - top - mh
            pad_left = left
            pad_right = W - left - mw
            return tf.pad(patch, [[0,0],[0,0],[pad_top, pad_bottom],[pad_left, pad_right]])

        out = tf.zeros_like(xft)

        if mh > 0 and mw > 0:
            # Build complex weights
            w00 = tf.complex(self.w00_r, self.w00_i)  # (mh,mw,C,C)
            w01 = tf.complex(self.w01_r, self.w01_i)
            w10 = tf.complex(self.w10_r, self.w10_i)
            w11 = tf.complex(self.w11_r, self.w11_i)

            # Top-left [0:mh, 0:mw]
            x00 = xft[:, :, :mh, :mw]                                    # (B,C,mh,mw)
            y00 = tf.einsum('bcij,ijco->boij', x00, w00)                 # (B,C,mh,mw)
            out += place(y00, 0, 0)

            # Top-right [0:mh, W-mw:W]
            x01 = xft[:, :, :mh, -mw:]
            y01 = tf.einsum('bcij,ijco->boij', x01, w01)
            out += place(y01, 0, tf.shape(x)[2] - mw)

            # Bottom-left [H-mh:H, 0:mw]
            x10 = xft[:, :, -mh:, :mw]
            y10 = tf.einsum('bcij,ijco->boij', x10, w10)
            out += place(y10, tf.shape(x)[1] - mh, 0)

            # Bottom-right [H-mh:H, W-mw:W]
            x11 = xft[:, :, -mh:, -mw:]
            y11 = tf.einsum('bcij,ijco->boij', x11, w11)
            out += place(y11, tf.shape(x)[1] - mh, tf.shape(x)[2] - mw)

        y = tf.signal.ifft2d(out)                                       # (B,C,H,W)
        y = tf.transpose(tf.math.real(y), [0, 2, 3, 1])                 # (B,H,W,C)

        # pointwise mixing
        y = y + self.w_point(x)
        return y


class FNO2DBlock(layers.Layer):
    """FNO-2D block: spectral mixing + residual + GELU."""
    def __init__(self, modes_h: int, modes_w: int, **kwargs):
        super().__init__(**kwargs)
        self.sconv = SpectralConv2D(modes_h, modes_w)
        self.norm = layers.LayerNormalization(axis=-1)

    def call(self, x):
        y = self.sconv(x)
        y = self.norm(y)
        y = tf.nn.gelu(y)
        return x + y


class ResidualConvBlock(layers.Layer):
    """Local conv refinement: (3x3 -> GELU -> 3x3) + residual."""
    def __init__(self, width: int, dilation: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(width, 3, padding="same", dilation_rate=dilation)
        self.conv2 = layers.Conv2D(width, 3, padding="same", dilation_rate=dilation)
        self.norm = layers.LayerNormalization(axis=-1)

    def call(self, x):
        y = self.conv1(x)
        y = tf.nn.gelu(y)
        y = self.conv2(y)
        y = self.norm(y)
        return x + y


# ------------------------------
# Model builder
# ------------------------------

def build_lp_fno_all_sides(
    H, W, Cin, Cout=2,
    side_width=64, side_depth=3, side_modes=8,
    cross_width=32,
    trunk_width=64, trunk_depth=3, trunk_modes=(8, 8),
    conv_blocks=2, profile_hidden=32
):
    inp = keras.Input(shape=(H, W, Cin))

    # Extract the four sides
    top   = layers.Lambda(lambda t: t[:, 0, :, :], name="side_top")(inp)    # (N, W, Cin)
    bottom= layers.Lambda(lambda t: t[:, -1, :, :], name="side_bottom")(inp)# (N, W, Cin)
    left  = layers.Lambda(lambda t: t[:, :, 0, :], name="side_left")(inp)   # (N, H, Cin)
    right = layers.Lambda(lambda t: t[:, :, -1, :], name="side_right")(inp) # (N, H, Cin)

    # Side encoders (share weights per orientation to reduce params, optional)
    enc_h = SideEncoder1D(side_width, side_depth, side_modes, name="enc_horizontal")  # for top/bottom
    enc_v = SideEncoder1D(side_width, side_depth, side_modes, name="enc_vertical")    # for left/right

    f_top = enc_h(top)          # (N, W, E)
    f_bot = enc_h(bottom)       # (N, W, E)
    f_lft = enc_v(left)         # (N, H, E)
    f_rgt = enc_v(right)        # (N, H, E)

    # Lifts
    lift_T = LiftHorizontal(H, side_width, hidden=profile_hidden, name="lift_top")
    lift_B = LiftHorizontal(H, side_width, hidden=profile_hidden, name="lift_bottom")
    lift_L = LiftVertical(W, side_width, hidden=profile_hidden, name="lift_left")
    lift_R = LiftVertical(W, side_width, hidden=profile_hidden, name="lift_right")

    w_T = lift_T(f_top)  # (N,H,W,E)
    w_B = lift_B(f_bot)
    w_L = lift_L(f_lft)
    w_R = lift_R(f_rgt)

    # # Cross term
    # cross = CrossTerm(cross_width, name="cross_term")
    # w_X = cross(f_top, f_bot, f_lft, f_rgt)  # (N,H,W,cross_width)

    # Fuse lifted fields
    # fused = layers.Concatenate(axis=-1, name="fuse_lifts")([w_T, w_B, w_L, w_R, w_X])  # (N,H,W,C_lift)
    fused = layers.Concatenate(axis=-1, name="fuse_lifts")([w_T, w_B, w_L, w_R])  # (N,H,W,C_lift)

    # Add coordinate channels
    fused = AddCoords2D(name="add_coords")(fused)  # (N,H,W,C_lift+2)

    # Project to trunk width
    x = layers.Conv2D(trunk_width, kernel_size=1, padding="same", name="pre_trunk")(fused)

    # 2D FNO trunk
    mh, mw = trunk_modes
    for i in range(trunk_depth):
        x = FNO2DBlock(mh, mw, name=f"fno2d_block_{i+1}")(x)

    # Local conv refinement
    for i in range(conv_blocks):
        x = ResidualConvBlock(trunk_width, dilation=1, name=f"local_conv_{i+1}")(x)

    # Head
    out = layers.Conv2D(Cout, kernel_size=1, padding="same", name="head")(x)

    model = keras.Model(inputs=inp, outputs=out, name="LPFNO_AllSides")
    return model
