import os
import tensorflow as tf
from hyperparams import Hyperparams as hp

def main():

    saved_model_dir = os.path.join(hp.logdir, 'SavedModel', "MobMatt-{}".format(hp.image_size))

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
                                           tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
                                          ]
    tflite_quant_mobmatt = converter.convert()
    
    tflite_model_dir = os.path.join(hp.logdir, 'TFLite')
    os.makedirs(tflite_model_dir, exist_ok = True)
    save_path = os.path.join(tflite_model_dir, "MobileMatt.tflite")
    with open(save_path, 'wb') as f: f.write(tflite_quant_mobmatt)
    
    print(f"TFLite exported to {save_path}")

if __name__ == '__main__':
    main()