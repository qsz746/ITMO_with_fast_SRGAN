# ITMO using SRGAN
We use SRGAN to directly learn mapping from SDR to HDR

# Setup and useful commands

```bash
# clone the repo in westgrid
git clone https://github.com/dingqingy/ITMO_with_fast_SRGAN.git

cd ITMO_with_fast_SRGAN

# copy pretrained VGG weights
mkdir pretrained
cd pretrained
cp ~/projects/def-panos/shared_itmo/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5 .

# go back and launch the evaluation script
cd ../
sbatch eval.slurm

# check job status
sq

# check job output: replace {job_id} with the corresponding job id
vim slurm-{job_id}.out

# if you are not sure about which output file is the most recent one
ls -lhtr # list files in reverse time order
```

# To contribute 
- Folk the repo
- clone your folk
- create a new branch
- implement your fixes or features
- push the branch to your remote
- fire a pull request
