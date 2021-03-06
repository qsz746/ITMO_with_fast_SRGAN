Project name: ITMO<br />
Team members : Sizhuo Qi, Jing Cao, Dingqing Yang, Nikola Cucuk<br />

The programming language used to develop the code Python and the version: python/3.6 <br />
The images for training and testing are in exr. form. <br />
Deep learning frameworks used: Keras/TensorFlow. <br />
The version of that framework used: For example, tensorflow==2.0.1 . <br />
The operating systems used for the development and their versions.(Windows/ MacOS Catalina Version 10.15.5). <br />
The requirements/dependencies and their versions that are needed to run the code: opencv-python==4.1.1.26, numpy==1.17.2 <br />
The name of the deep learning network used for the development VGG19. <br />
Link to the third-party codes used in the project (for example from GitHub) and brief summary of that code. https://github.com/qsz746/ITMO_with_fast_SRGAN <br />

**pretrained**<br />
To store VGG19 network.<br />

**dataloader.py**<br />
Data Loader prepares the data object for training.

**models**<br />
The code will output a discriminator/generator pair every 50 iterations under directory models/. <br />

**train_itmo**<br />
Batch file to train the network<br />

**infer_itmo**<br />
Batch inference allows us to generate predictions on a batch of samples such as testing images.

**testingimages**<br />
Images for testing.<br />

**requirements.txt**<br />
tensorflow==2.0.1<br />
opencv-python==4.1.1.26<br />
numpy==1.17.2<br />

We used external computing services WestGrid, it makes any difference to run them on the conventional PCs and laptops. Our project is running on beluga which belongs to westGrid. Thus, we can use the virtual environment and GPUs to run the training process. In another word, our own laptops don't have the capacity and ability to trian such a large dataset. <br />


# Before you run:

1. mkdir pretrained in your ITMO_with_fast_SRGAN directory and you have to put VGG vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5 in "pretrained" directory as we did before.
2. You have to modify all the sizhuoqi (line30, line35, line36) in train_itmo, use you own $USER. sizhuoqi is my own $USER.
3. You have to modify all the sihzuoqi in infer_itmo.
4. mkdir output in your ITMO_with_fast_SRGAN directory, so you can see output images when the infer_itmo job is done.


## For training
The code will output a discriminator/generator pair every 50 iterations under directory models/. <br />
`$ sbatch train_itmo`

## Download log files and use local tensorboard to visualize log files.
`$ tensorboard --logdir=logs`

## For testing, use pretrained model and see the output images
We need to give the path of the pretrained generater.h5 under "--gen" in the infer_itmo. And then run: <br />

`$ sbatch infer_itmo`



# To contribute 
- Folk the repo
- clone your folk
- create a new branch
- implement your fixes or features
- push the branch to your remote
- fire a pull request
