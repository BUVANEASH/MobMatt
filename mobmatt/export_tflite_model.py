import os
import numpy as np
import tensorflow as tf
from trainer import Trainer
from hyperparams import Hyperparams as hp

class MobileMatt(tf.keras.Model):

    def __init__(self, mobmatt, image_size,
                 name = 'MobileMatt', **kwargs):
        super(MobileMatt, self).__init__(name = name, **kwargs)
        self.mobmatt = mobmatt
        self.image_size = image_size
        
    @tf.function
    def call(self, inputs):
        
        x = inputs
        _, h, w, c = inputs.shape.as_list()
        
        x = tf.image.resize(x, size = [self.image_size, self.image_size], method = 'bilinear')

        x = self.mobmatt(x, training = False)

        x = tf.image.resize(x, size = [h, w], method = 'bilinear')

        x = tf.cast(x * 255, dtype = tf.uint8)

        x = tf.concat([inputs, x], axis = -1)

        return x

def main():

    modeltrainer = Trainer()

    modeltrainer.update(hp.__dict__)

    params = "{0:25} | {1:25}"
    for k,v in modeltrainer.__dict__.items():
        if isinstance(v,(str,int,float,np.ndarray)):
            print(params.format(k,v))

    modeltrainer.build()

    modeltrainer.load_ckpts(partial=True)

    model = modeltrainer.mobmatt

    input_data = tf.ones((1,hp.image_size,hp.image_size,3), dtype = tf.uint8)

    _ = model(input_data)

    mobmatt = MobileMatt(mobmatt = model, image_size = hp.image_size, name = 'MobileMatt')

    _ = mobmatt(input_data)

    tflite_model_dir = os.path.join(hp.logdir, 'TFLite')
    os.makedirs(tflite_model_dir, exist_ok = True)

    mobmatt_call_fn = mobmatt.call.get_concrete_function(tf.TensorSpec(shape=[1,hp.image_size,hp.image_size,3], dtype=tf.uint8, name = 'input'))

    converter = tf.lite.TFLiteConverter.from_concrete_functions([mobmatt_call_fn], mobmatt)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    
    tflite_quant_mobmatt = converter.convert()
    
    save_path = os.path.join(tflite_model_dir, "MobileMatt.tflite")
    with open(save_path, 'wb') as f: f.write(tflite_quant_mobmatt)
    
    print(f"TFLite exported to {save_path}")

if __name__ == '__main__':
    main()