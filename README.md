## Playing with SMPLify-X

<p align="center">
    <img src="construction.gif", width="600">
</p>

This is a fork of the [[SMPLify-X repository](https://github.com/vchoutas/smplify-x)]. I only made it to share some code modifications I've made to that code, all the credits go to the original authors.

Read also the medium post:
https://medium.com/@apofeniaco/p3d-body-detection-with-smplify-x-ced6c38871df

SMPLify-X can be quite tricky to understand, as it has lots of different configurations and long pieces of code. The modifications on this repo aim towards letting run SMPLify-X with zero configuration and providing better insights on how does it work.

  * Separates configuration parsing from running code to keep it more easy to read
  * Allows to run human body detection on a single image and show the results once it has finishing.
  * Allows to run the 2D keypoint detection on the fly with OpenPose, without having to run it first to generate the JSON output files.
  * Shows how the SMPLify-X optimizer is working, rendering the crafting of the body model just on top of the processed image.
  
## Install

  * To Install, first follow SMPLify-X installation instructions. You have to download the body model used from the project website, registering and accepting the terms for its use license. Once confirmed the mail and logged into the site, on the Downloads section you will find the “SMPL-X Model” and the “VPoser: Variational Human Pose Prior”, download and extract them somewhere on your disk. 
  * You need to install also OpenPose and its Python binding, follow instructions at [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).
  * Write the path to your OpenPose model's folder on params[“model_folder”] in openpose_wrapper.py.
  * On this repository folder, create folders run/images and run/keypoints. They stay empty, are used to trick SMPLify-X original implementation. Then run:
 
```Shell
python3 smplifyx/easy_run.py --config cfg_files/fit_smplx.yaml  \
    --data_folder run \
    --output_folder out \
    --visualize="True" \
    --model_folder $MODEL_FOLDER \
    --vposer_ckpt $VPOSER_FOLDER \
    --input_media $IMAGE_PATH 
```

Where $MODEL_FOLDER is the path where you downloaded and extracted the SMPL-X Model, $VPOSER_MODEL is the path for the VPoser, and $IMAGE_PATH is the path to the image on which you want to detect bodies.


