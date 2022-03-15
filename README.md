# MobMatt

## Model
---

A Tensorflow implementation of [`Efficient Models for Real-time Person Segmentation
on Mobile Phones`](https://eurasip.org/Proceedings/Eusipco/Eusipco2021/pdfs/0000651.pdf) with last layer modified for predicting alpha mask. It is a UNet-like network architecture based on MobileNetV3.

### MobMatt Architecture (from original paper)

![MobMatt](/documents/images/MobMatt.png)

### MobMatt Encoder

For the encoder part in MobMatt, used **`MobileNetV3Small`** from **`tf.keras.applications`** with the preprocessing included.

## Python Requirements
---

- tqdm
- numpy
- matplotlib
- opencv-python
- tensorflow-addons
- tensorflow-gpu==2.8.0

## Dataset
---

Used [aisegmentcom-matting-human-datasets](https://www.kaggle.com/laurentmih/aisegmentcom-matting-human-datasets) from kaggle.

Download, extract and move the folders *`'aisegmentcom-matting-human-datasets/matting_human_half/clip_img'`* and *`'aisegmentcom-matting-human-datasets/matting_human_half/matting'`* into *`clip_img`* and *`matting`* respectively as shown below

	MobMatt\
	 └─ data\
	     └─ raw\
             ├─ clip_img\
             │   ├─ 1803XXXXXX
             │   ├─ ...
             │   └─ 1803XXXXXX\
             │       ├─ clip_XXXXXXXX
             │       ├─  ...
             │       └─ clip_XXXXXXXX\
             │           ├─ 1803XXXXXX_XXXXXXXX.jpg
             │           ├─  ...
             │           └─ 1803XXXXXX_XXXXXXXX.jpg
             └─ matting\
                 ├─ 1803XXXXXX
                 ├─ ...
                 └─ 1803XXXXXX\
                     ├─ matting_XXXXXXXX
                     ├─  ...
                     └─ matting_XXXXXXXX\
                         ├─ 1803XXXXXX_XXXXXXXX.png
                         ├─  ...
                         └─ 1803XXXXXX_XXXXXXXX.png

## Training
---

Modify the `hyperparams.py` script's *data_path* attributes appropriately to point to the correct dataset folder and *logdir* for saving checkpoints and tensorboard logs.

> **RUN**

	python train.py

Monitor the training using tensorboard with the tensorboard log file under the *`tensorboard`* folder under the log directory 

***Observe whether the loss and SAD values are converging.***

Once the training is done, the checkpoints can be found in the *`ckpts`* folder in log directory

	MobMatt\
	 └─ results\
	     └─ models\
	    	 ├─ ckpts\
	    	 │	 ├─ checkpoint
	    	 │	 ├─ ckpt-25000.data-00000-of-00001
	    	 │	 └─ ckpt-25000.index
	    	 └─ tensorboard\
	    	 	 └─ events.out.tfevents.x.x.x.x.v2

## Testing

Basic inference testing on images and create a **`RGBA`**(*.png*) output images

> **RUN**

	python test.py -i <Input Image or Input Images Dir> -o <Output dir>

## Exporting Model
---

To Export a TF SaveModel Format

> **RUN**

	python export_tf_model.py

To Export a TFLite Model

> **RUN**

	python export_tflite_model.py

Exported Models can be found in the following directories

    MobMatt\
	 └─ results\
	     └─ models\
             ├─ ckpts
             ├─ SavedModel\
             │	 └─ MobMatt-224
             ├─ tensorboard
             └─ TFLite\
             	 └─ MobMatt-224.tflite

## ToDo

- [x] Create Model
- [x] Train Model
- [x] Test Model
- [x] Export TFLite Model
- [ ] Test TFLite Model

## References

1. [Efficient Models for Real-time Person Segmentation
on Mobile Phones](https://eurasip.org/Proceedings/Eusipco/Eusipco2021/pdfs/0000651.pdf)
2. [Searching for MobileNetV3
](https://arxiv.org/pdf/1905.02244.pdf)
3. [Deep Image Matting](https://arxiv.org/pdf/1703.03872.pdf)