import tensorflow as tf
from keras.callbacks import TensorBoard

class BatchTensorBoard(TensorBoard):
    def __init__(self, log_period=1, **kwargs):
        super().__init__(**kwargs)
        self.log_period = log_period
        self.batch_counter = 0
    
    def on_batch_end(self, batch, logs=None):
        if self.batch_counter % self.log_period == 0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.batch_counter)
            self.writer.flush()
            
        self.batch_counter += 1
        
        super().on_batch_end(batch, logs)
        
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(self.batch_counter, logs)