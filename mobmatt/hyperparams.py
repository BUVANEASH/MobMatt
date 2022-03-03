import os

class hyperparameters():

    def __init__(self):
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
        self.batch_size = 64
        self.test_split = 0.10

        # Iteration
        self.num_iteration = 25000

        # Logging
        self.log_step = 10
        self.save_step = 1000
        self.summary_step = 100
        self.freeze_encoder = True

        # Dataset
        self.data_path = r"..\data"
        self.raw_data = os.path.join(self.data_path,'raw')
        self.clip_image_path = os.path.join(self.raw_data, 'clip_img')
        self.matting_image_path = os.path.join(self.raw_data, 'matting')

        # logdir
        self.logdir = r"..\results\models"

    def update(self,newdata):
        for key,value in newdata.items():
            setattr(self,key,value)

Hyperparams = hyperparameters()