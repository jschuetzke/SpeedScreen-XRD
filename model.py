from tensorflow import keras  # noqa: F401
from keras import layers, optimizers, regularizers, constraints, Model
from model_utils import Custom_BCE1, Custom_BCE2


def prospect_identification(
    input_size=2200,
    ks=17,
    conv_layers=4,
    opt=None,
    lr=1e-3,
):
    input_layer = layers.Input(shape=(input_size, 1), name="input")
    x = input_layer
    for i in range(conv_layers):
        pad = "same" if i else "valid"
        x = layers.Conv1D(32, ks, padding=pad, activation="gelu")(x)
        x = layers.MaxPooling1D(pool_size=3, strides=2, padding="valid")(x)
    gbl1 = layers.GlobalAveragePooling1D()(x)
    pm = layers.Permute((2, 1))(x)
    gbl2 = layers.GlobalMaxPooling1D()(pm)
    flat = layers.Concatenate(name="flat")([gbl1, gbl2])
    x = layers.BatchNormalization(momentum=0.7, scale=False, center=False)(flat)
    x = layers.Activation("relu")(x)
    out1 = layers.Dense(
        1,
        activation="sigmoid",
        kernel_regularizer=regularizers.L1L2(1e-3, 1e-4),
        kernel_constraint=constraints.NonNeg(),
        name="empty_structure",
    )(flat)
    out2 = layers.Dense(
        1,
        activation="sigmoid",
        kernel_constraint=constraints.NonNeg(),
        name="single_multi",
    )(x)
    model = Model(inputs=input_layer, outputs=[out1, out2])

    if opt is None:
        opt = optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=opt, loss=[Custom_BCE1(), Custom_BCE2()])
    return model
