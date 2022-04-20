import cv2
import numpy as np
import tensorflow as tf
from trainer import Trainer
from hyperparams import Hyperparams as hp

def main():

    model = Trainer()

    model.update(hp.__dict__)

    params = "{0:25} | {1:25}"
    for k,v in model.__dict__.items():
        if isinstance(v,(str,int,float,np.ndarray)):
            print(params.format(k,v))

    model.build()

    model.load_ckpts(partial=True)

    vid = cv2.VideoCapture(0)   

    # kernel for eroding ther predicted alpha mask
    kernel = np.ones((5,5), np.uint8)     
    
    while(True):
        
        ret, image = vid.read()

        if ret:
            
            h,w,_ = image.shape
            _image = tf.image.resize(image[...,::-1], size = (hp.image_size,hp.image_size), method = 'bilinear')                
            alpha = model.mobmatt(_image[tf.newaxis,...], training = False)[0]
            alpha = tf.image.resize(alpha, size = (h,w), method = 'bilinear').numpy()
            
            # applying erosion to alpha mask
            alpha = cv2.erode(alpha, kernel, iterations=1)
            # masking portrait
            output_image = np.array(image*alpha[...,np.newaxis], dtype = np.uint8)

            cv2.imshow('output', output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
    vid.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()