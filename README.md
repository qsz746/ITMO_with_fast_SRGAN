Before you run:

1. You have to put VGG vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5 in pretrained as we did before.
2. You have to modify all the sizhuoqi (line30, line35, line36) in train_itmo, use you own $USER. sizhuoqi is my own $USER.
3. You have to modify all the sihzuoqi in infer_itmo.


1. To train
$ sbatch train_itmo

2.  Download log files and use local tensorboard to visualize log files.
$ tensorboard --logdir=logs

3. Use pretrained model and see the output images
The code will output a discriminator/generator pair every 50 iterations under directory models/. You have to run the infer_itmo after the trained weight is generated, since you need to give the path of the generater.h5 under "--gen" in the infer_itmo.

$ sbatch infer_itmo

# To contribute 
- Folk the repo
- clone your folk
- create a new branch
- implement your fixes or features
- push the branch to your remote
- fire a pull request
