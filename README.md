[![DOI](https://zenodo.org/badge/doi/10.1364/BOE.528579.svg)](https://doi.org/10.1364/BOE.528579)

### Intro:
MOSS-DDPG is short for Multi-observation Single-step deep deterministic policy gradient, a sensorless adaptive optics image metric optimization method based on the deep deterministic policy gradient algorithm.

MOSS-DDPG optimizes the image quality by predicting the wavefront error formed by Zernike coefficients with 2*N+1 image acquisitions, where N is the number of Zernike modes used.

The MOSS-DDPG implementation code is a PyTorch-based simulation environment.

### Files:

**main.py:** main body for MOSS-DDPG training. 

**DSAO_env.py:** deep sensorless adaptive optics environment, including DDPG environment and image formation simulation codes for a confocal scanning laser ophthalmoscope.

**Networks.py:** actor-network and critic-network definition for MOSS-DDPG.

**noise.py:** exploration noise added during training.

**memory.py:** memory for DDPG training.

**train.py:** code for updating network weights given training data.

**critic_dataset.py:** a data loader for critic network training data management.

**model_test.py:** examine the performances of trained models with random aberration tests.

**rms_wfe_vis.py:** visualize the performance evaluation results of RMS wavefront error.

**metric_coefficient_curve.py** plots the image sharpness metric response versus Zernike mode coefficients for individual modes.

**training_visualization.py:** visualize training sessions.

**functions.py** some functions.

### Core packages:
pytorch, cupy, numpy, zernike (for Zernike polynomial generation https://github.com/jacopoantonello/zernike)

### An example of whole process - training and validation:
1. select target images and put into **Test Images** folder.
2. run **main.py**, training will start and be recorded in "Training History/date/time + training specs/records.txt".
3. change the file path variable in **training_visualization.py** to view the training process.
4. change the file path variable in **model_test.py** and run random tests, results will be saved in the same file path.
5. change the file path variable in **rms_wfe_vis.py** to visualize the random test results of the RMS wavefront error.
