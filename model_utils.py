import tensorflow as tf


class Custom_BCE1(tf.keras.losses.BinaryCrossentropy):
    # y_true -> classes 0,1,2
    # convert 1,2 -> 1
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.cast(y_true, tf.bool), tf.float32)
        return super().__call__(y_true, y_pred)


class Custom_BCE2(tf.keras.losses.BinaryCrossentropy):
    # y_true -> classes 0,1,2
    # convert 1,2 -> 0,1
    # set sample_weight of class 0 to 0, rest 1
    def __call__(self, y_true, y_pred, sample_weight=None):
        # sample weight 0 for class 0
        sample_weights = tf.cast(tf.cast(y_true, tf.bool), tf.float32)
        y_true = tf.math.abs((y_true - 1))
        return super().__call__(y_true, y_pred, sample_weight=sample_weights)


def multiout_acc(pred1, pred2, y):
    pred1 = tf.convert_to_tensor(pred1, dtype=tf.float32)
    pred2 = tf.convert_to_tensor(pred2, dtype=tf.float32)

    pred1 = tf.math.round(pred1)
    pred2 = tf.math.round(pred2) + 1
    pred = pred1 * pred2
    pred = tf.gather(pred, indices=0, axis=1)
    pred = tf.cast(tf.math.round(pred), tf.int32)
    pred = tf.one_hot(pred, 3)
    hit = tf.keras.metrics.categorical_accuracy(y, pred)
    acc = tf.math.reduce_sum(hit) / tf.cast(tf.size(hit), tf.float32)
    return acc


class CatAcc(tf.keras.callbacks.Callback):
    def __init__(self, x, y, name=""):
        self.x = x
        self.y = tf.one_hot(y, 3)
        self.name = name
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = {} or logs
        _, res1, res2 = self.model.predict(self.x, verbose=0)
        acc = multiout_acc(res1, res2, self.y)
        prefix = "" if self.name == "" else self.name + "_"
        logs[prefix + "accuracy"] = acc
        super().on_epoch_end(epoch, logs)
