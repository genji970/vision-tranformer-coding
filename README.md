# vision-tranformer-coding
coding trying 

This is on going. So there can be some mistakes.

At main.py model runs.

At Data_loading.py, only subset of CIFAR10 dataset will be loaded. 

At input_processing.py, there are two functions
  1) at image_to_patches, images divided into patches
     # shape : B , N , embedding vector_size = patch_size^2 * channel_num
  2) at preprocessing, cls_tocken and positional embedding will be added to image.
     
At multihead_attention.py, attention process will be runned.

At Encoder.py, encoder structure will be runned.(several encoding can be easily runned if you just add more encoders)
