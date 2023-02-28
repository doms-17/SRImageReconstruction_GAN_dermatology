# Real-ESRGAN Architecture for Dermoscopy Images: User Guide
This guide provides step-by-step instructions on how to use the Real-ESRGAN architecture on dermoscopy images.

## Installation
To use Real-ESRGAN, follow these steps:

1. Open the Real-ESRGAN repository: https://github.com/xinntao/Real-ESRGAN
2. Install all required settings by following the instructions provided in the README file.

## Training
To train the Real-ESRGAN architecture, follow these steps:

1. Read the README file on the Real-ESRGAN repository: https://github.com/xinntao/Real-ESRGAN/blob/master/docs/Training.md
2. Prepare the paired dataset using the Preprocessing Pipeline using the **dermasr package**.

## To fine-tune using a pre-trained model:
3. Download the desired weights from the Real-ESRGAN model zoo: https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md
4. Put the **finetune_realesrgan_derma_pairdata.yml** file inside the *option* folder of the Real-ESRGAN repository.
5. On the command line, enter: `python realesrgan/train.py -opt options/finetune_realesrgan_derma_pairdata.yml --auto_resume`

## Inferencing
To perform inferencing on images using the trained model, follow these steps:

1. Put the weights of the trained model inside the weights folder of the Real-ESRGAN repository.
2. Rename it dermaRealESRGAN_x2 or change the name of the trained model inside the script: **inference_realesrgan_derma.py**. It is important that the name and parameters coincide with the ones used in the training (which are set inside the **finetune_realesrgan_derma_pairdata.yml** file). 
For example:
 ```   
    # here the name of the model must coincide with the name of the weight (.pth file) obtained at the end of the training phase. Weights must be stored in the weights folder of the Real-ESRGAN repository:
    if args.model_name in ['dermaRealESRGAN_x2']:  # if you change the name here, be sure to rename accordingly also the weight file (.pth)
        # parameters here must coincide with the ones chosen in the configuration (.yml) file:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)  
        netscale = 2
 ```
3. Open the bash in the main path of the Real-ESRGAN repository and enter on the command line: `python inference_realesrgan_derma.py -n dermaRealESRGAN_x2 -i path_which_contains_images_to_inference -o path_where_to_save_inferenced_images --outscale 2`

**NOTE** that the training was set to create 1024 Super-Resolution images from 512 Low-Resolution counterparts in this thesis work. The settings and situation may vary, but only x2 and x4 types of tasks are tested successfully. Inside the finetune_realesrgan_derma_pairdata.yml used for the training part and the inference_realesrgan_derma.py used for the inferencing part, it is possible to customize parameters and change the settings accordingly to the type of task to perform.