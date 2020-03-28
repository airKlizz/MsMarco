import tensorflow as tf

class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, num_classes, name='confusion_matrix', **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.matrix = self.add_weight(name='matrix', shape=(num_classes, num_classes), initializer='zeros', dtype=tf.dtypes.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        labels = tf.argmax(y_true, -1)
        predictions = tf.argmax(y_pred, -1)
        step_matrix = tf.math.confusion_matrix(labels, predictions, num_classes=self.num_classes)
        self.matrix.assign_add(step_matrix)

    def result(self):
        return self.matrix

    def reset_states(self):
        self.matrix.assign(tf.zeros([self.num_classes, self.num_classes], tf.int32))