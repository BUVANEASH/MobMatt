import os
import cv2
import glob
import random
import datetime
import numpy as np
import tensorflow as tf
from typing import Union
import matplotlib.pyplot as plt
from contextlib import nullcontext
from dataload import DataLoad
from mobmatt import MobMatt

class Trainer(DataLoad):
    """
    Model trainer class
    """
    def __init__(self):
        super(Trainer, self).__init__()
        
        self.gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for _device in self.gpu_devices:
            tf.config.experimental.set_memory_growth(_device, True)
            print(f"TF using {_device}")
        tf.config.set_soft_device_placement(True)
        if len(self.gpu_devices):
            self.use_mixed_precision = True
            self.device = "/gpu:0"
        else:
            self.use_mixed_precision = False
            self.device = "/cpu:0"

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Dataset
        self.image_size = 224
        self.image_vertical_prob = 0.5
        self.image_brightness = 0.2
        self.image_contrast = [0.8,1.2]
        self.image_translate_prob = 0.3
        self.image_translate_x = [-20,20]
        self.image_translate_y = [-20,20]
        self.image_rotate_prob = 0.3
        self.image_rotate_angle = 0.3

        # Training
        self.learning_rate = 1e-4
        self.batch_size = 8
        self.test_split = 0.10
        self.freeze_encoder = True
    
    def build(self):
        if self.use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
        
        exp_decay_lr = tf.keras.optimizers.schedules.ExponentialDecay(self.learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(exp_decay_lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        if self.use_mixed_precision:
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)
        
        self.mobmatt = MobMatt(image_size = self.image_size, 
                               freeze_encoder = self.freeze_encoder, 
                               name = "MobMatt")

        self.checkpoint = tf.train.Checkpoint(step = self.global_step,
                                              optimizer = self.optimizer,
                                              mobmatt = self.mobmatt)

        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, os.path.join(self.logdir,"ckpts"),
                                                             checkpoint_name='ckpt',
                                                             max_to_keep=3)

        self.built = True
    
    def sum_of_absolute_difference(self, target : tf.Tensor, predicted : tf.Tensor) -> tf.Tensor:
        return tf.math.reduce_mean(tf.math.abs(target - predicted))

    def _difference(self, target : tf.Tensor, predicted : tf.Tensor) -> tf.Tensor:
        return tf.math.reduce_mean(tf.math.sqrt(tf.math.abs(target - predicted)**2 + 1e-5**2))

    def compute_loss(self, target_alpha : tf.Tensor, predicted_alpha : tf.Tensor, image : tf.Tensor) -> tf.Tensor:
        alpha_prediction_loss = self._difference(target_alpha, predicted_alpha)
        target_composite = image * target_alpha
        predicted_composite = image * predicted_alpha
        composite_loss = self._difference(target_composite, predicted_composite)
        loss = 0.5 * alpha_prediction_loss + 0.5 * composite_loss
        return loss
    
    def load_ckpts(self, partial : bool = False):

        if partial:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint).expect_partial()
        else:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            print("Restored from {}".format(self.checkpoint_manager.latest_checkpoint))
        else:
            print("No checkpoints, Initializing from scratch.")
    
    def train(self):

        log_dir = os.path.join(self.logdir, "tensorboard", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)

        writer = tf.summary.create_file_writer(log_dir)

        self.load_ckpts()
        self.get_dataset()
        train_dataloader = self.create_dataloader(self.train_set).__iter__()
        test_dataloader = self.create_dataloader(self.test_set).__iter__()

        with writer.as_default():
            with tf.summary.record_if(True):
                while int(self.checkpoint.step) <= self.num_iteration:
                    batch_images, batch_masks = next(train_dataloader)

                    # Train Step
                    start_time = datetime.datetime.now()
                    predicted_alpha, loss, sad_value = self.train_step(batch_images, batch_masks, training = True)
                    duration = datetime.datetime.now() - start_time
                    
                    if int(self.checkpoint.step) % self.summary_step == 0:
                        test_batch_images, test_batch_masks = next(test_dataloader)
                        test_predicted_alpha, test_loss, test_sad_value = self.train_step(test_batch_images, test_batch_masks, training = False)
                        
                        # Write Summary
                        self.write_summary(loss, sad_value, batch_images, batch_masks, predicted_alpha, prefix = 'train')
                        self.write_summary(test_loss, test_sad_value, test_batch_images, test_batch_masks, test_predicted_alpha, prefix = 'test')

                    self.checkpoint.step.assign_add(1)

                    if int(self.checkpoint.step) % self.save_step == 0:
                        _ = self.checkpoint_manager.save(checkpoint_number = int(self.checkpoint.step))
                    
                    if int(self.checkpoint.step) % self.log_step == 0:
                        print("{0} step {1}, loss = {2}, sad = {3}, ({4} examples/sec; {5} sec/batch)".
                            format(datetime.datetime.now(), int(self.checkpoint.step), loss, sad_value, (self.batch_size/duration.total_seconds()), duration.total_seconds()))

        print("Training Done.")

    def write_summary(self, loss : tf.Tensor, sad : tf.Tensor, images : tf.Tensor, masks : tf.Tensor, alpha : tf.Tensor, prefix : str = ''):
        tf.summary.scalar(prefix + "_loss", loss, step=int(self.checkpoint.step))
        tf.summary.scalar(prefix + "_sad", sad, step=int(self.checkpoint.step))
        
        tf.summary.image(prefix + "_image", tf.cast(images, dtype = tf.uint8), step=int(self.checkpoint.step))
        tf.summary.image(prefix + "_pred_alpha", alpha, step=int(self.checkpoint.step))
        tf.summary.image(prefix + "_mask", masks, step=int(self.checkpoint.step))

        composite = tf.cast(images * alpha, dtype = tf.uint8)
        tf.summary.image(prefix + "_composite", composite, step=int(self.checkpoint.step))
    
    @tf.function
    def train_step(self, images : tf.Tensor, masks : tf.Tensor, training : bool = True) -> tuple[tf.Tensor]:
        with tf.device(self.device):
            cnxtmngr = tf.GradientTape() if training else nullcontext()
            with cnxtmngr as tape:
                predicted_alpha = self.mobmatt(images, training = training)
                
                loss = self.compute_loss(masks, predicted_alpha, images)
                sad_value = self.sum_of_absolute_difference(masks, predicted_alpha)
                
                trainable_variables = self.mobmatt.trainable_variables
            
            if training:
                grads = tape.gradient(loss, trainable_variables)
                if self.use_mixed_precision:
                    grads = self.optimizer.get_unscaled_gradients(grads)
                grads = [(tf.clip_by_value(grad, -1, 1)) for grad in grads]

                self.optimizer.apply_gradients(zip(grads, trainable_variables))
        
        return predicted_alpha, loss, sad_value

    def update(self, newdata : dict):
        for key,value in newdata.items():
            setattr(self,key,value)