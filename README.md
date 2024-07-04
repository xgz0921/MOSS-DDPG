MOSS-DDPG is short for Multi-observation Single-step deep deterministic policy gradient, a sensorless adaptive optics image metric optimization method based on the deep deterministic policy gradient algorithm.

MOSS-DDPG optimizes the image quality by predicting the wavefront error formed by Zernike coefficients with 2*N+1 image acquisitions, where N is the number of Zernike modes used.

The MOSS-DDPG implementation code is a PyTorch-based simulation environment.

Files:

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
