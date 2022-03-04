import os
import numpy as np
import tensorflow as tf
from trainer import Trainer
from hyperparams import Hyperparams as hp

def main():

    modeltrainer = Trainer()

    modeltrainer.update(hp.__dict__)

    params = "{0:25} | {1:25}"
    for k,v in modeltrainer.__dict__.items():
        if isinstance(v,(str,int,float,np.ndarray)):
            print(params.format(k,v))

    modeltrainer.build()

    modeltrainer.load_ckpts(partial=True)

    mobmatt = modeltrainer.mobmatt

    input_data = tf.ones((1,hp.image_size,hp.image_size,3), dtype = tf.uint8)

    _ = mobmatt(input_data)

    tf_model_dir = os.path.join(hp.logdir, 'SavedModel')
    export_path = os.path.join(tf_model_dir, "MobMatt-{}".format(hp.image_size))
    
    mobmatt.save(export_path, include_optimizer=False, save_format='tf')

    print(f"TF SavedModel exported to {export_path}")

if __name__ == '__main__':
    main()