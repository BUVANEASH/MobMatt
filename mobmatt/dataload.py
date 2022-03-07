import os
import glob
import random
import numpy as np
import tensorflow as tf
from typing import Generator
import tensorflow_addons as tfa

class DataLoad():
    """
    Dataset loader class
    """
    def __init__(self):
        pass
    
    def get_dataset(self):
        self.matting_image_list = sorted(glob.glob(os.path.join(self.matting_image_path,'*/*/*.png')))
        total_len = len(self.matting_image_list)
        self.train_set = self.matting_image_list[:int(total_len*(1-self.test_split))]
        self.test_set = self.matting_image_list[int(total_len*(1-self.test_split)):]
        train_len = len(self.train_set)
        test_len = len(self.test_set)
        print(f"Total {total_len} -----> Train {train_len} | Test {test_len}")

    def map_fn(self, image_path: str, mask_path: str) -> tuple[tf.Tensor]:
        """
        Args:
            image_path: The RGB image path.
            mask_path: The RGBA mask image path.

        Returns:
            The augmented image, mask pair
        """
        # Read Image and Mask
        image = tf.io.decode_jpeg(tf.io.read_file(image_path))
        mask = tf.io.decode_png(tf.io.read_file(mask_path), channels = 4)[...,3:]
        # Resize
        image = tf.image.resize(image,(self.image_size,self.image_size), method = 'bilinear')
        mask = tf.image.resize(mask,(self.image_size,self.image_size), method = 'bilinear')
        # MobileNetV3 includes image normalization inbuilt
        # # Normalize
        image = tf.cast(image, tf.uint8)
        mask = tf.cast(mask, tf.float32) * (1. / 255)

        # Augmentation
        brightness = tf.random.uniform((), -self.image_brightness, self.image_brightness)
        image = tf.image.adjust_brightness(image, brightness)

        contrast = tf.random.uniform((), self.image_contrast[0], self.image_contrast[1])
        image = tf.image.adjust_contrast(image, contrast)

        if tf.random.uniform((), 0, 1) > self.image_vertical_prob:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)

        if tf.random.uniform((), 0, 1) > self.image_translate_prob:
            tx = tf.random.uniform((), self.image_translate_x[0], self.image_translate_x[1], dtype = tf.int32)
            ty = tf.random.uniform((), self.image_translate_y[0], self.image_translate_y[1], dtype = tf.int32)
            image = tfa.image.translate(image, translations = [tx,ty], interpolation = 'bilinear', fill_mode = 'nearest')
            mask = tfa.image.translate(mask, translations = [tx,ty], interpolation = 'bilinear', fill_mode = 'nearest')

        if tf.random.uniform((), 0, 1) > self.image_rotate_prob:
            ra = tf.random.uniform((), -self.image_rotate_angle, self.image_rotate_angle)
            image = tfa.image.rotate(image, angles = ra, interpolation = 'bilinear', fill_mode = 'nearest')
            mask = tfa.image.rotate(mask, angles = ra, interpolation = 'bilinear', fill_mode = 'nearest')
        
        image = tf.cast(image, dtype = tf.float32)

        return image, mask
    
    def create_generator(self, img_set: list[str]) -> Generator[tuple[str], None, None]:
        """
        Args:
            img_set: The RGBA image paths set
            
        Yield:
            The augmented image, mask pair
        """
        def generator():
            for mask_path in img_set: 
                image_path = os.path.join(self.clip_image_path,mask_path.split("\\",4)[-1].replace('matting','clip').replace('.png','.jpg'))        
                yield image_path, mask_path
            
        return generator
    
    def create_dataloader(self, image_set: list[str]) -> tf.data.Dataset:
        """
        Args:
            gen: image set generator
            
        Yield:
            The image set dataloader
        """
        dataset = tf.data.Dataset.from_generator(self.create_generator(image_set),
                                                 output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),
                                                                   tf.TensorSpec(shape=(), dtype=tf.string)))
        dataset = dataset.map(lambda i, m : self.map_fn(i,m))
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.shuffle(buffer_size = 100)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)
        
        return dataset