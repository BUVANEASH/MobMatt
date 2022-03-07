import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from trainer import Trainer
from hyperparams import Hyperparams as hp

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input','-i', type=str,
                        help='Input Image or Input Image Dir')
    parser.add_argument('--output','-o', type=str, default = r"..\results\outputs",
                        help='Output Dir')

    args = parser.parse_args()

    model = Trainer()

    model.update(hp.__dict__)

    params = "{0:25} | {1:25}"
    for k,v in model.__dict__.items():
        if isinstance(v,(str,int,float,np.ndarray)):
            print(params.format(k,v))

    model.build()

    model.load_ckpts(partial=True)

    image_paths = []
    if os.path.isfile(args.input):
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        for ext in ['png','jpg','jpeg']:
            image_paths += sorted(glob.glob(os.path.join(args.input,f'*.{ext}')))
    
    for image_path in tqdm(image_paths):
        
        image = tf.cast(cv2.imread(image_path), dtype = tf.uint8)
        h,w,c = image.shape.as_list()
        _image = tf.image.resize(image, size = (hp.image_size,hp.image_size), method = 'bilinear')[...,::-1]
        
        alpha = model.mobmatt(_image[tf.newaxis,...], training = False)[0]
        alpha = tf.image.resize(alpha, size = (h,w), method = 'bilinear')
        alpha = tf.cast(alpha * 255, dtype = tf.uint8)
        
        name = os.path.basename(image_path).split('.')[0]
        output_path = os.path.join(args.output, f'{name}.png')
        
        output_image = tf.concat([image,alpha], axis = -1).numpy()
        cv2.imwrite(output_path, output_image)
        
if __name__ == '__main__':
    main()