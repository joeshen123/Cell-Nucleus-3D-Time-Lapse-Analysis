import numpy as np

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [16, 12]

from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_3d
from skimage.morphology import remove_small_objects, binary_closing, ball, disk, erosion, dilation   # function for post-processing (size filter)
from aicssegmentation.core.MO_threshold import MO
from aicssegmentation.core.utils import get_middle_frame, get_3dseed_from_mid_frame
from skimage import transform, measure, filters
import h5py
from tqdm import tqdm
from pandas import DataFrame
import warnings
from skimage.segmentation import watershed, clear_border
from skimage.morphology import dilation, ball
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
from aicssegmentation.core.utils import hole_filling
import trackpy as tp
from tkinter import simpledialog
from tkinter import filedialog
from tkinter import messagebox
import tkinter as tk
from colorama import Fore
import napari
import mplcursors
#Ignore warnings issued by skimage through conversion to uint8
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)

# Select file for import
root = tk.Tk()
root.withdraw()

my_filetypes = [('all files', '.*'),('Movie files', '.nd2')]

file_name = filedialog.askopenfilename(title='Please Select a Movie', filetypes = my_filetypes)

print(file_name)
f = h5py.File(file_name, 'r')

image = f['561 Channel'][:]

#Make a mask to clear border
mask = np.ones(image[0].shape)

for n in range(mask.shape[0]):
   im_slice = mask[n,:,:]
   im_slice[0,:] = 0
   im_slice[:,0] = 0
   im_slice[:,-1] = 0
   im_slice[-1,:] = 0

mask = mask.astype(np.bool)

# Make a function to get middle frame based on segmented area
def get_middle_frame_area(labelled_image_stack):
    max_area = 0
    max_n = 0
    for z in range(labelled_image_stack.shape[0]):
       img_slice = labelled_image_stack[z,:,:]
       area = np.count_nonzero(img_slice)
       
       if area >= max_area:
          max_area = area
          max_n = z
    
    return max_n


# Initiate the cell segmentation class

class cell_segment:
    def __init__ (self, Time_lapse_image, intensity_scaling_param = [40000], gaussian_smoothing_sigma = 1 ):
        self.image = Time_lapse_image.copy()
        self.Time_pts = Time_lapse_image.shape[0]
        
        self.normalization_param = intensity_scaling_param
        self.smooth_param = gaussian_smoothing_sigma
        
        self.structure_img_smooth = np.zeros(self.image.shape, dtype = np.float64)
        self.structure_img = np.zeros(self.image.shape, dtype = np.float64)
        self.segmented_object_image = np.zeros(self.image.shape, dtype = np.uint8)
        
        self.watershed_map_list = np.zeros(self.image.shape, dtype = np.uint8)
        self.seed_map_list = np.zeros(self.image.shape, dtype = np.uint8)
        
        
    # define a function to apply normalization and smooth on Time lapse images
    def img_norm_smooth (self):
        pb = tqdm(range(self.Time_pts), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET))
        for t in pb:
            pb.set_description("Image Normalization and Gaussian Smooth")
            img = self.image[t].copy()
            self.structure_img[t] = intensity_normalization(img, scaling_param=self.normalization_param)
            self.structure_img_smooth[t] = image_smoothing_gaussian_3d(self.structure_img[t], sigma=self.smooth_param)
        
   # define a function to apply Masked Object thresholding followed by seed-based watershed to each time point
    def MO_threshold_Time (self):
        pb = tqdm(range(self.Time_pts), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET))
        for t in pb:
            pb.set_description("Masked Object Thresholding + Watershed")
            bw, object_for_debug = MO(self.structure_img_smooth[t], global_thresh_method='ave', object_minArea=10000, return_object=True,extra_criteria=True)
            bw = hole_filling(bw, 1, 5000, fill_2d=True)
            mid_z = get_middle_frame(bw, method='intensity')
            bw_mid_z = bw[mid_z,:,:]
            seed = get_3dseed_from_mid_frame(bw_mid_z, bw.shape, mid_z, hole_min=3000,bg_seed=False)
            watershed_map = -1*distance_transform_edt(bw)
            seg = watershed(watershed_map, markers=label(seed), mask=bw, watershed_line=True)
            seg = clear_border(seg, mask = mask)
            
            seg = remove_small_objects(seg>0, min_size=10000, connectivity=1, in_place=False)
            seg = hole_filling(seg, 1, 40000, fill_2d=True)
        
            
            final_seg = label(seg)
            
            self.segmented_object_image[t] = final_seg
            self.watershed_map_list[t] = watershed_map
            self.seed_map_list[t] = seed
        

# Initiate the cell attribute class

class cell_attribute:
    def __init__ (self, segmented_image_seq, intensity_image_stack, smooth_sigma = 1.0):
        self.labeled_stack = segmented_image_seq
        self.positions_table = None
        self.intensity_image_stack_raw = intensity_image_stack
        self.intensity_image_stack = np.zeros(intensity_image_stack.shape)
        self.labeled_stack_modify = self.labeled_stack.copy()
        self.smooth_sigma = smooth_sigma
        self.mid_slice_stack = np.zeros(intensity_image_stack.shape)
    # Function to 3D gaussian smooth intesnity image stack
    def intensity_stack_smooth (self):
        pb = tqdm(range(self.intensity_image_stack.shape[0]), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.CYAN, Fore.RESET))
        for t in pb:
          pb.set_description("Smooth intensity image")
          img = self.intensity_image_stack_raw[t].copy()
          self.intensity_image_stack[t] = image_smoothing_gaussian_3d(img, sigma=self.smooth_sigma)
            
    # function to create pandas table of cell attributes without tracking info
    def create_table_regions(self):

        positions = []
        
        pb = tqdm(range(self.labeled_stack.shape[0]), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
        for n in pb:
            pb.set_description("Generate Statistic Table")
            labeled_slice = self.labeled_stack[n]
    
            for region in measure.regionprops(labeled_slice):
                position = []

                z_pos = region.centroid[0]
                y_row = region.centroid[1]
                x_col = region.centroid[2]

                volume = region.area * (0.19*0.19*1)

                nucleus_image = labeled_slice == region.label
                mid_z = get_middle_frame_area(nucleus_image)
                
                mid_nucleus_image = nucleus_image[mid_z,:,:]
                
                segmented_image_shell= np.logical_xor(mid_nucleus_image,erosion(mid_nucleus_image, selem=disk(6)))
                 
                
                label = region.label
                intensity_single = self.intensity_image_stack[n]
                intensity_image = intensity_single[mid_z]
                intensity_median = np.median(intensity_image[segmented_image_shell==True])

                intensity_background = np.median(self.intensity_image_stack[n][int(mid_z),int(y_row),int(x_col)])

                intensity_median_ratio = intensity_median / intensity_background

                position.append(x_col)
                position.append(y_row)
                position.append(z_pos)
                position.append(int(n))
                position.append(label)


                position.append(volume)
                #position.append(surface_area_actual)
                position.append(intensity_median_ratio)

                positions.append(position)
                
                self.mid_slice_stack[n,mid_z,:,:] += segmented_image_shell
                self.mid_slice_stack[n,mid_z,int(y_row),int(x_col)] = 1
                
        self.positions_table  = DataFrame(positions, columns = ['x','y','z',"frame",'label','volume', 'median intensity ratio'])

 
# Cell segmentation
track = cell_segment(image)
track.img_norm_smooth()
track.MO_threshold_Time()

 
# Statistic generation 
Segment_tracker = cell_attribute(track.segmented_object_image,image)
Segment_tracker.intensity_stack_smooth()
Segment_tracker.create_table_regions() 
 
# Tracking
# Link cells between frames
Labelled_table = tp.link_df(Segment_tracker.positions_table, 15,adaptive_stop=1, adaptive_step=2, memory=3,pos_columns=['x', 'y', 'z'])

#Filter out tracks with low number of frames
Labelled_table = tp.filter_stubs(Labelled_table, 60)

#Normalize binding to the start
Labelled_table['normalized intensity ratio'] = Labelled_table['median intensity ratio']

all_labels = Labelled_table.particle.unique()

for label in all_labels:
   Labelled_table['normalized intensity ratio'][Labelled_table.particle ==label] = Labelled_table['normalized intensity ratio'][Labelled_table.particle ==label]/Labelled_table['median intensity ratio'][Labelled_table.particle ==label][0]

 
#Visualizations
with napari.gui_qt():
 viewer = napari.Viewer()
 viewer.add_image(image, scale = [1,7,1,1])
 viewer.add_image(Segment_tracker.mid_slice_stack,name='Mid Plane', scale = [1,7,1,1])
 viewer.add_image(track.watershed_map_list,name='Distance Map', scale = [1,7,1,1])
 viewer.add_image(track.segmented_object_image>0 ,name='Segmented Object', scale = [1,7,1,1])
 peaks = np.nonzero(track.seed_map_list)
 viewer.add_points(np.array(peaks).T, name='peaks', size=5, face_color='red',scale=[1,7,1,1])

# Plot volume for all objects (cells)
def vol_plotter(df):
   all_l = df.particle.unique()

   for label in all_l:
     df_subset = df[df.particle == label]
     plt.plot(df_subset["frame"],df_subset["volume"],label = str(label))
   
   plt.tight_layout()
   mplcursors.cursor(highlight=True,hover=True)
   plt.show()
 
vol_plotter(Labelled_table)
# Plot normalized intensity for all objects (cells)
def intensity_plotter(df):
   all_l = df.particle.unique()

   for label in all_l:
    df_subset = df[df.particle == label]
    plt.plot(df_subset["frame"],df_subset["normalized intensity ratio"],label = str(label))


   plt.tight_layout()
   mplcursors.cursor(highlight=True,hover=True)  
   plt.show()
 
intensity_plotter(Labelled_table)

#Delete tracks based on user's input
del_answer = messagebox.askyesnocancel("Question","Do you want to delete some measurements?")

while del_answer == True:
    delete_answer = simpledialog.askinteger("Input", "What number do you want to delete? ",
                                                 parent=root,
                                                 minvalue=0, maxvalue=100)
        
    Labelled_table = Labelled_table[Labelled_table.particle != delete_answer]
    
    vol_plotter(Labelled_table)
    intensity_plotter(Labelled_table)

    del_answer = messagebox.askyesnocancel("Question","Do you want to delete more measurements?")

#Export the segmentation result and statistic table
File_save_names = '.'.join(file_name.split(".")[:-1])
save_name='{File_Name}_analysis'.format(File_Name = File_save_names)

csv_save_name = save_name + '.csv'

Labelled_table.to_csv(csv_save_name)

seg_save_name='{File_Name}_segmentation_result_3D.hdf5'.format(File_Name = File_save_names)

with h5py.File(seg_save_name, "w") as f:
      f.create_dataset('Segmentation_Binary_Result', data = track.segmented_object_image, compression = 'gzip')
      f.create_dataset('Mid_Plane_Selection', data = Segment_tracker.mid_slice_stack, compression = 'gzip')
      f.create_dataset('Distance Map', data = track.watershed_map_list, compression = 'gzip')
      f.create_dataset('Seed Map', data = track.seed_map_list, compression = 'gzip')


    

    