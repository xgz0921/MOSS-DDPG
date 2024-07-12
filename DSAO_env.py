### Script for MOSS-DDPG simulation environment
# Make sure target images are put into Test Images directory before running.
### @author : Guozheng Xu
### @date   : 2024-07-11
############################################################################
# Dependencies
import gym, os
import numpy as np
import scipy 
import scipy 
from scipy.ndimage import zoom
import cupy as cp
from cupyx.scipy.signal import convolve2d as convolve2d_gpu
import random
import matplotlib.pyplot as plt
import matplotlib.image as imm

from zernike import RZern #For generating Zernike polynomials. 
"""
This package 'https://github.com/jacopoantonello/zernike' implements the methods described in the following publication:
Antonello, J., & Verhaegen, M. (2015). Modal-based phase retrieval for adaptive optics.
J. Opt. Soc. Am. A, 32(6), 1160-1170. doi:10.1364/JOSAA.32.001160
Available at: https://opg.optica.org/josaa/abstract.cfm?URI=josaa-32-6-1160
"""


co_range = 0.15 #Random Zernike coefficient range in micrometers
obs_bias = 0.5 # Observation bias in micrometers
cr_range = 1.0 
wfres = 100 #Wavefront resolution
n_modes = 12 #Number of target modes
n_modes_all = 25 #Number of all modes, including higher order noise modes

#%% Set Zernike coefficient ranges in micrometers.
#Coefficient ranges for training.
# the 6th and 7th radial order Zernike coefficients are simulated for higher order noise.
co_ranges_train = np.array([0.15, 0.15, 0.15,\
             0.15, 0.15, 0.15, 0.15,\
             .15, .15, .15, .15, .15,\
            0.025,0.025,0.025,0.025,0.025,0.025,
            0.025,0.025,0.025,0.025,0.025,0.025,0.025])*1
#Decreasing coefficient ranges to simulate mouse eye
co_ranges_decreasing = np.array([0.2, 0.2, 0.2,\
             0.15, 0.15, 0.15, 0.15,\
             .1, .1, .1, .1, .1,\
            0.025,0.025,0.025,0.025,0.025,0.025,
            0.025,0.025,0.025,0.025,0.025,0.025,0.025])*1

#Uniform coefficient ranges for testing, without the effect of higher order modes.
co_ranges_clean = np.array([0.15, 0.15, 0.15,\
             0.15, 0.15, 0.15, 0.15,\
             .15, .15, .15, .15, .15,\
            0,0,0,0,0,0,
            0,0,0,0,0,0,0])*1

#Wavefront scale calibration. The scale of the Zernike coefficients for simulating the real system to generate the same wavefront phase shape as the real system.
coefficient_scale = 1000/488 #488 nm laser used, change as needed.
co_scales = np.array([coefficient_scale for _ in range(n_modes_all)]) 

#%% Generate tuples containing mode indices and coefficient ranges & Other useful variables. 
crDM_all = tuple((-cr_range,cr_range,i)for i in range(4,n_modes_all+4))
crDM_obs = tuple((-obs_bias,obs_bias,i)for i in range(4,n_modes+4))
crDM_obs_signs_only = tuple((-1,1,i)for i in range(4,n_modes+4)) #This is for generating observation matrix
abDM_all = tuple((-co_ranges_decreasing[i-4],co_ranges_decreasing[i-4],i)for i in range(4,n_modes_all+4))
abDM = tuple((-co_range,co_range,i)for i in range(4,n_modes+4))

rda = .0,1.0 #Reward dynamic range [0,1]
iobs = 2*len(abDM)+1 #number of observations

#%% Load target images.
def load_images(directory_path):
    '''
    directory_path: The path for all target images. 'Test Images' by default.
    '''
    def process_image(file_path):
        '''
        file_path: the file path to individual image file.
        '''
        img = imm.imread(file_path).astype('float')
        if len(img.shape) == 2:  # Grayscale image
            gray_img = img
        else:  # Color image, convert to grayscale
            gray_img = img[:,:,0]*0.2989 + img[:,:,1]*0.587 + img[:,:,2]*0.114
        return np.array(gray_img)
    
    # List to store processed images
    processed_images = []
    
    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(directory_path, filename)
            processed_img = process_image(file_path)
            processed_images.append(processed_img)
    
    return processed_images
            
trgtims = load_images('Test Images') #Load target images from the default directory 'Test Images'.
img_size = 100 #Define the image size to be convolved with the PSF
#%%
  
class MOSSDDPG_Env(gym.Env):
    class cr: #Correction DM
        class sim:pass
    class ab: #Aberration DM
        class sim:pass
    class dq:pass #Data acquisition
    class sim:pass #Simulated SAO ingredients
    
    def __init__(self):
        #Observation & Action Space Definition
        self.observation_space = gym.spaces.Box(np.array([np.full(iobs,mnob) for mnob in [rda[0]]+[ob[0] for ob in crDM_obs]]).T,
												np.array([np.full(iobs,mxob) for mxob in [rda[1]]+[ob[1] for ob in crDM_obs]]).T)

        self.action_space = gym.spaces.Box(np.array([ob[0] for ob in crDM_obs]),
									 np.array([ob[1] for ob in crDM_obs]))
        
        self.mxi = self.mni = 0
        self.cr.c = np.zeros(len(crDM_all))
        self.cr.ar = crDM_all
        self.ab.c = np.zeros(len(abDM_all))
        self.ab.ar = abDM_all
        
        #The following two parameters represent the scale for the pupil size - from a diameter of "wfres" pixels to (wfres/scaling_factor) pixels. Smaller pupil size result in
        #smaller NA and larger Airy Disk Diameter. The two values represent relative ADD sizes for the two paths of the confocal SLO system, can be adjusted based on specific system parameters. 
        self.scaling_i = 2 #Factor for scaling illumination path PSF size.
        self.scaling_c = 3 #Factor for scaling detection path PSF size.

        # Define coordinates for wavefront, both illumination and detection path.
        self.xpr_i = np.linspace(-self.scaling_i,self.scaling_i,wfres) 
        self.ypr_i = self.xpr_i
        self.xpr_c = np.linspace(-self.scaling_c,self.scaling_c,wfres)
        self.ypr_c = self.xpr_c

        # Zernike modes
        self.cart_i = RZern(7)
        self.cart_c = RZern(7)
        self.xv_i,self.yv_i = np.meshgrid(self.xpr_i,self.ypr_i)
        self.xv_c,self.yv_c = np.meshgrid(self.xpr_c,self.ypr_c)
        self.cart_i.make_cart_grid(self.xv_i, self.yv_i)
        self.cart_c.make_cart_grid(self.xv_c, self.yv_c)
        
        # Unit wavefront lists for all modes
        self.sim.wflst_i = [0]*int(self.cart_i.nk)
        self.sim.wflst_c = [0]*int(self.cart_c.nk)

        #Initialization
        self.SimInit() 
        
        # Define pinhole (detection fiber) parameters. Uniform circle for simplicity.
        # Determine suitable size based on the relative size between detection path ADD and fiber core diameter.
        fiber_size  = 0.07 # This is an arbitrary parameter for the fiber size. The diameter of the unit circle is wfres*fiber_size.
        self.fiber = np.zeros((wfres,wfres))
        
        # Generate a unit circle representing the fiber (approximation), diameter = wfres*fiber_size
        self.fiber[(self.xv_c/self.scaling_c)**2+(self.yv_c/self.scaling_c)**2<=fiber_size**2] = 1

        # Observation matrix components definition
        self.obs_r = np.zeros((iobs,1))
        self.obs_c = np.zeros((iobs,1))
        
        # Other parameters
        self.trtem = 0 # Count the steps already performed for each episode.
        self.rststp = 1 # Steps before reset, single step per episode for this scenario.
        self.step_count = 0 # Count total steps experienced
        self.episode_count = 0 # Count number of episodes experienced
        self.rwd_record = [] # Record of rewards during training
        self.wfe_record = [] # Record of wavefront errors during training
        
        self.trgtim = np.random.uniform(0,255,(img_size,img_size)) # Initialize a random array as the target image
        
    def step(self,action):
        
        self.trtem += 1                                 
        self.step_count+=1
        corrections = np.clip(np.array(action),-0.2,0.2) #Clip the actions to be within effective range.
        self.rwd_temp = self.crSet(corrections) #Apply the actor network's predictions to the DM, correct wavefront aberration, and calculate the reward.

        reward = np.power(self.rwd_temp/self.flat,1)

        rst = (self.trtem >= self.rststp) #End episode when steps exceed the range (only 1 step here since rststp=0)
        
        if rst:
            self.rwd_record.append(reward) #Record the rewards
            
        return [], reward, rst, {}
    
    def reset(self, training = True, ab_mode = 'Decreasing',obs_bias = obs_bias, new_ab = True):
        '''
        Reset function of DDPG. 
        training: determine if it is training or not. If training, use uniformly distributed Zernike coefficients to form random aberration.
        ab_mode: random aberration mode for testing, including "decreasing", "normal","uniform", and "nonoise"
        '''
        self.trtem = 0 #Set the number of steps performed per episode to zero
        
        #Different aberration generation schemes, depending on training or not. (customizable)
        if new_ab:
            #select a random part of a random image from the test images
            self.max_intensity = 1 # Set the image intensity normalizer to 1 before acquiring the reference perfect image.
            trgtim_selected = random.choices(trgtims)[0]
            trgtim_sz = trgtim_selected.shape[0]
            start_x = random.randint(0, int(trgtim_sz-img_size))
            start_y = random.randint(0, int(trgtim_sz-img_size))
            self.trgtim = trgtim_selected[start_x:start_x + img_size, start_y:start_y + img_size]

            self.ab.c = np.zeros(n_modes_all)
            self.cr.c = np.zeros(n_modes_all)
            self.abSet()
            self.flat_img = self.crSet(save_fig = True) #Acquire the reference perfect image (flat wavefront)
            self.max_intensity = np.max(self.flat_img.flatten()) # Get max image pixel intensity
            self.flat_img = self.flat_img/self.max_intensity #Normalize perfect image intensity from 0 to 1
            self.flat = sum(self.flat_img.flatten()**2) #Acquire the perfect image metric (flat wavefront)
            
        if training:
            if new_ab:
            #Uniform aberration amount for all modes is used for training.
                aberrations = np.zeros(n_modes_all)
                self.ab.c[:n_modes] = np.array([np.round(np.random.uniform(-co_ranges_train[md[2]-4],co_ranges_train[md[2]-4]),3) for md in abDM]) 
                self.ab.c[n_modes:] = np.clip(np.array([np.round(np.random.normal(0,0.4*co_ranges_train[md[2]-4]),3) for md in abDM_all[n_modes:]]),-0.025,0.025)
        else:
            if new_ab:
                #Generate higher order mode noise.
                self.ab.c[n_modes:] = np.clip(np.array([np.round(np.random.normal(0,0.4*co_ranges_train[md[2]-4]),3) for md in abDM_all[n_modes:]]),-0.025,0.025)
                
                if ab_mode == 'decreasing':#Gaussian distributed coefficient values with decreasing limits
                    self.ab.c[:n_modes] = np.array([np.round(np.clip(np.random.normal(0,0.4*co_ranges_decreasing[md[2]-4]),-0.2,0.2),3) for md in abDM])
                    
                elif ab_mode == 'normal':#Gaussian distributed coefficient values with uniform limits
                    self.ab.c[:n_modes] = np.array([np.round(np.random.normal(0,0.4*co_ranges_train[md[2]-4]),3) for md in abDM])
                
                elif ab_mode == 'nonoise':#Uniform distributed coefficient values with uniform limits, higher order mode noise removed.
                    self.ab.c[:n_modes] = np.array([np.round(np.random.normal(0,0.4*co_ranges_decreasing[md[2]-4]),3) for md in abDM_all])
                    self.ab.c[n_modes:] = np.array([0 for md in abDM_all[n_modes:]])
                    
                elif ab_mode == 'uniform':#Uniformly distributed coefficients, same as training.
                    self.ab.c[:n_modes] = np.array([np.round(np.random.uniform(-co_ranges_train[md[2]-4],co_ranges_train[md[2]-4]),3) for md in abDM]) 
        self.abSet() #Set aberrations on the DM
        #Acquire the metrics for 2N+1 observations: self.obs_r
        crDM_rst = tuple((-obs_bias,obs_bias,i)for i in range(4,n_modes+4)) #Form a tuple to generate observation matrix.
        
        # Get the image metrics: self.obs_r
        [self.obs_r, temp] = zip(*[([self.crSet(md)], md) for md in [np.array([(ar[0]+ar[1])/2 for ar in crDM_rst])]+ \
									  [np.insert(np.delete([(ar[0]+ar[1])/2 for ar in crDM_rst], i), i, n) \
									   for i in range(len(crDM_rst)) for n in crDM_rst[i][0:2]]])
        
        # Form the right side of the observation matrix :self.obs_c
        [temp, self.obs_c] = zip(*[([0], md) for md in [np.array([(ar[0]+ar[1])/2 for ar in crDM_obs_signs_only])]+ \
									  [np.insert(np.delete([(ar[0]+ar[1])/2 for ar in crDM_obs_signs_only], i), i, n) \
									   for i in range(len(crDM_obs_signs_only)) for n in crDM_obs_signs_only[i][0:2]]])
        
        #Find the maximum of acquired metrics for 2N+1 observations for normalization.
        self.obs_r = np.array(self.obs_r)
        self.mxi = np.amax(self.obs_r)
        self.mni = 0 #np.amin(self.obs_r) #Can also normalize with (x-min)/(max-min). If self.mni = 0, metrics are normalized as x/max

        self.episode_count += 1 # Record a new episode.
        
        return np.concatenate(((self.obs_r-self.mni)/(self.mxi-self.mni),self.obs_c),axis = 1)
    
    def SimInit(self):
        #Initialization of Zernike polynomials.
        def setzer(n):
            c_i = np.zeros(self.cart_i.nk)
            c_i[n-1] = 1
            self.sim.wflst_i[n] = self.cart_i.eval_grid(c_i,matrix=True)
            
            c_c = np.zeros(self.cart_c.nk)
            c_c[n-1] = 1
            self.sim.wflst_c[n] = self.cart_c.eval_grid(c_c,matrix=True)
            
            return
        [setzer(int(md)) for md in list(dict.fromkeys([md[2] for md in crDM_all+abDM_all]))]
        return
    
    def abSet(self):
        #Set wavefront error, assuming illumination path and collection path have the same wavefront pattern. 
        self.ab.sim.wf_i = sum([self.sim.wflst_i[self.ab.ar[md][2]]*self.ab.c[md]*co_scales[md] for md in range(len(self.ab.ar))]) #illumination path wavefront
        self.ab.sim.wf_c = sum([self.sim.wflst_c[self.ab.ar[md][2]]*self.ab.c[md]*co_scales[md] for md in range(len(self.ab.ar))]) #collection (detection) path wavefront
        return
    
    def crSet(self,*mdar,save_fig = False):
        #Set wavefront correction signal (Zernike coefficients) to the DM
        if mdar: #If given a target array of coefficients
            self.cr.c[0:n_modes] = np.squeeze(mdar) #Set the first 12 modes to desired values
            self.cr.c[n_modes:] = np.zeros(n_modes_all-n_modes) # Do not correct for the noise modes.
            
        return self.ZK(self.cr,save_fig=save_fig)

    def conv_fiber(self,psf,fiber):
        #Convolution of PSF with Fiber
        psf_cuda = cp.asarray(psf)
        fiber_cuda = cp.asarray(fiber)
        out = convolve2d_gpu(psf_cuda, fiber_cuda,mode = 'same')
        return out.get()

    def ZK(self,dm,save_fig = False):
        '''
        Reward acquisition (Image generation) based on current aberration and correction DM shapes.
        '''
        dm.sim.wf_i = sum([self.sim.wflst_i[dm.ar[md][2]]*dm.c[md]*co_scales[md] for md in range(len(dm.ar))]) #Wavefront superposition, illumination path
        dm.sim.wf_c = sum([self.sim.wflst_c[dm.ar[md][2]]*dm.c[md]*co_scales[md] for md in range(len(dm.ar))]) #Wavefront superposition, collection path
        psi_i = np.exp(-1j*(self.cr.sim.wf_i+self.ab.sim.wf_i)) #PSI
        psi_c = np.exp(-1j*(self.cr.sim.wf_c+self.ab.sim.wf_c)) #PSI
        psi_i[np.isnan(psi_i)]=0
        psi_c[np.isnan(psi_c)]=0
       
        #Point spread function generation. For fast convolution, only central [50,50] is used. 
        h_r = 0.25
        h_i = abs(np.fft.fftshift(np.fft.fft2(psi_i)))**2
        h_i = h_i[int(0.5*wfres)-int(h_r*wfres):int(0.5*wfres)+int(h_r*wfres),int(0.5*wfres)-int(h_r*wfres):int(0.5*wfres)+int(h_r*wfres)]
        h_c = abs(np.fft.fftshift(np.fft.fft2(psi_c)))**2
        h_c = h_c[int(0.5*wfres)-int(h_r*wfres):int(0.5*wfres)+int(h_r*wfres),int(0.5*wfres)-int(h_r*wfres):int(0.5*wfres)+int(h_r*wfres)]
        h_c_fiber = self.conv_fiber(h_c,self.fiber[47:53,47:53])#Convolution result between pinhole and detection path PSF
        g = h_i * h_c_fiber #Effective PSF 'g' for the entire system. 
        g_cuda = cp.asarray(g) 
        trgtim_cuda = cp.asarray(self.trgtim)
        fig_out = convolve2d_gpu(trgtim_cuda,g_cuda,mode='same').get()/self.max_intensity #Get image by convolution between target image and effective PSF.
        #The division number is to normalize the output image intensity based on the maximum intensity from a perfect image.

        self.dq.rdar = fig_out #Data acquired and saved to data acquisition class.

        # Photon and detector noise can be added, but will slow down training process. Since transfer learning is required for in-situ application and simulation is just a
        # transitional process, the noise addition can be omitted. Performances of the models are consistent with (a reasonable amount) or without noise addition. This can be tuned for specific systems.
        # In-situ noise is expected to be learned by transfer learning.
        '''
        if self.max_intensity != 1: #reference image intensity for normalization should be generated. if not, do not add noise
            self.dq.rdar = np.clip(np.random.poisson(self.dq.rdar*255)/255 + scipy.stats.truncnorm(0, 1, 0, .02).rvs(size = self.dq.rdar.shape), 0, 1) #Add gaussian and poisson noise 
        '''

        if save_fig: #If this is true, return the actual figure, otherwise, return the metric
            return self.dq.rdar
        #Return the sharpness metric.
        return sum(self.dq.rdar.flatten()**2)
    
    def expand_dims(self,In):
        # Add an additional dimension to an array.
        m = np.expand_dims(In, axis=0)
    	    
        return m

# An example for using the simulation environment to get an image from the system with defined aberration and correction coefficients.
'''
env = MOSSDDPG_Env()
env.reset(new_ab=True) #Reset to get random image and perfect reference
env.ab.c = np.zeros(n_modes_all) #Define the aberration coefficients
env.abSet() #Set the aberration
env.cr.c = np.zeros(n_modes_all) #Define the correction coefficients 
fig = env.crSet(save_fig=True) #Apply correction coefficients and acquire the image
plt.imshow(fig) #plot the image
plt.show()
'''
