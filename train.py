import os
import numpy as np
import tensorflow as tf
from tensorflow import keras  # noqa: F401
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model import prospect_identification
from model_utils import CatAcc


# enable memory growth instead of blocking whole VRAM
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], "GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

system_name = "spinel"
step_size = 0.015

# import data
xt = np.load(os.path.join(system_name, "x_train.npy"))
yt = np.load(os.path.join(system_name, "y_train.npy"))

xv = np.load(os.path.join(system_name, "x_val.npy"))
yv = np.load(os.path.join(system_name, "y_val.npy"))

model = prospect_identification(input_size=2667, ks=17)

callbacks = [
    EarlyStopping(patience=25, verbose=1, restore_best_weights=True, min_delta=0.0001),
    ReduceLROnPlateau(patience=15, verbose=1),
    CatAcc(xt, yt, name='train'),
    CatAcc(xv, yv, name='val'),
]

model.fit(
    xt, yt, 128,
    epochs=1000,
    verbose=2,
    callbacks=callbacks,
    validation_data=(xv, yv),
)
tf.keras.models.save_model(
    model, os.path.join(system_name, f"model_{system_name}.h5"), save_format="h5"
)
