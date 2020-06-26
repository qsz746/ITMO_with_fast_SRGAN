The programming language used to develop the code Python and the version: python/3.6 
Deep learning frameworks used: Keras/TensorFlow. 
load nixpkgs/16.09  intel/2018.3  cuda/10.1 cudnn/7.6.5
The version of that framework used: For example, tensorflow==2.0.1 .
The operating systems used for the development and their versions.(Windows/ MacOS Catalina Version 10.15.5).
The requirements/dependencies and their versions that are needed to run the code: opencv-python==4.1.1.26, numpy==1.17.2
The name of the deep learning network used for the development VGG19.
Link to the third-party codes used in the project (for example from GitHub) and brief summary of that code.
We used external computing services WestGrid, explain if it makes any difference to run them on the conventional PCs and laptops.


# Before you run:

1. mkdir pretrained in your ITMO_with_fast_SRGAN directory and you have to put VGG vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5 in "pretrained" directory as we did before.
2. You have to modify all the sizhuoqi (line30, line35, line36) in train_itmo, use you own $USER. sizhuoqi is my own $USER.
3. You have to modify all the sihzuoqi in infer_itmo.
4. mkdir output in your ITMO_with_fast_SRGAN directory, so you can see output images when the infer_itmo job is done.

operating system to use (including version), which data and their format and everything else that you think is essential for a third party to install and run your code to reproduce your results.

## For training
The code will output a discriminator/generator pair every 50 iterations under directory models/.
`$ sbatch train_itmo`

## Download log files and use local tensorboard to visualize log files.
`$ tensorboard --logdir=logs`

## For testing, use pretrained model and see the output images
We need to give the path of the pretrained generater.h5 under "--gen" in the infer_itmo. And then run:

`$ sbatch infer_itmo`



# To contribute 
- Folk the repo
- clone your folk
- create a new branch
- implement your fixes or features
- push the branch to your remote
- fire a pull request
