from keras.callbacks import Callback
from sklearn.metrics import roc_curve, auc

def RocAucMetric(Callback):
    def __init__(self, validation_generator, **kwargs):
        super().__init__(**kwargs)
        self.validation_generator = validation_generator
    
    def on_epoch_end(self, epoch, logs={}):
        inputs, true = next(self.validation_generator)
        pred = self.model.predict(inputs)
        fpr, tpr, threshold = roc_curve(true, pred)
        roc_auc = auc(fpr, tpr)
        print(f'val_roc_auc: {roc_auc}')
        
        super().on_epoch_end(epoch, logs)