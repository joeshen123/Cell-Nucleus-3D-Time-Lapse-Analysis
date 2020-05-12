from nd2reader.reader import ND2Reader
import numpy as np
import time
from tkinter import ttk
from tkinter import simpledialog
from tkinter import filedialog
import tkinter as tk
from skimage.external import tifffile
import matplotlib.pyplot as plt
from skimage import io
import cv2
import warnings
import h5py


#Ignore warnings issued by skimage through conversion to uint8
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)

# Use tkinter to interactively select files to import
root = tk.Tk()
root.withdraw()

my_filetypes = [('all files', '.*'),('Movie files', '.nd2')]

Image_Stack_Path = filedialog.askopenfilename(title='Please Select a Movie', filetypes = my_filetypes)

print (Image_Stack_Path)
# Define a function to convert time series of ND2 images to a numpy list of Max Intensity Projection
# images.

def Z_Stack_Images_Extractor(address, fields_of_view):
   Image_Sequence = ND2Reader(address)
   print(Image_Sequence.metadata["channels"])
   time_series = Image_Sequence.sizes['t']
   z_stack = Image_Sequence.sizes['z']
   
   Stack_561 = []
   Stack_488 = []

   n = 0

   # create progress bar
   windows = tk.Tk()
   windows.title("Extracting All Time points and Planes")
   s = ttk.Style(windows)
   
   s.layout("LabeledProgressbar",
         [('LabeledProgressbar.trough',
           {'children': [('LabeledProgressbar.pbar',
                          {'side': 'left', 'sticky': 'ns'}),
                         ("LabeledProgressbar.label",
                          {"sticky": ""})],
           'sticky': 'nswe'})])

   progress = ttk.Progressbar(windows, orient = 'horizontal', length = 1000, mode = 'determinate',style = "LabeledProgressbar")
   s.configure("LabeledProgressbar", text="0 / %d     ", troughcolor ='white', background='red')

   progress.grid()
   progress.pack(side=tk.TOP)
   progress['maximum'] = time_series

   progress['value'] = n
   
   progress.update_idletasks()

   for time in range(time_series):
     z_stack_561 = [] 
     z_stack_488 = []
     for z_slice in range(z_stack):
        slice_561 = Image_Sequence.get_frame_2D(c=0, t=time, z=z_slice, v=fields_of_view)
        slice_488 = Image_Sequence.get_frame_2D(c=1, t=time, z=z_slice, v=fields_of_view)
        z_stack_561.append(slice_561)
        z_stack_488.append(slice_488)

     z_stack_561 = np.array(z_stack_561)
     z_stack_488 = np.array(z_stack_488)

     Stack_561.append(z_stack_561)

     Stack_488.append(z_stack_488)
     
     n+=1
     progress['value'] = n

     s.configure("LabeledProgressbar", text='%d / %d   ' %(n, time_series))
     progress.update()

   Stack_561 = np.array(Stack_561)
   Stack_488 = np.array(Stack_488)

   progress.destroy()

   return (Stack_561, Stack_488)


Image_Sequence = ND2Reader(Image_Stack_Path)
FOV_list = Image_Sequence.metadata['fields_of_view']

Image_list_561 = []
Image_list_488 = []

for fov in range(len(FOV_list)):
   Images_561, Images_488 = Z_Stack_Images_Extractor(Image_Stack_Path,fields_of_view=fov)
   Image_list_561.append(Images_561)
   Image_list_488.append(Images_488)
   
'''
# Use parrallel computing to speedup saving files

def hdf5_save_file(file_name, num):
   GUV_Image_Name='{file_save_name}_{n}.hdf5'.format(file_save_name = file_name, n = num + 1)

   GUV_Images = Image_list_561[num]
   Image_Intensity = Image_list_488[num]
   
   with h5py.File(GUV_Image_Name, "w") as f:
      f.create_dataset('488 Channel', data = Image_Intensity, compression = 'gzip')
      f.create_dataset('561 Channel', data = GUV_Images, compression = 'gzip')
   


File_save_names = '.'.join(Image_Stack_Path.split(".")[:-1])

#Parallel(n_jobs=20)(delayed(hdf5_save_file)(File_save_names,n) for n in range(len(FOV_list)))
'''
File_save_names = '.'.join(Image_Stack_Path.split(".")[:-1])

for n in range(len(FOV_list)):
   Cell_Image_Name='{File_Name}_{num}.hdf5'.format(File_Name = File_save_names, num = n + 1)
   
   Images_561_Final = Image_list_561[n]
   Images_488_Final = Image_list_488[n]

   with h5py.File(Cell_Image_Name, "w") as f:
      f.create_dataset('561 Channel', data = Images_561_Final, compression = 'gzip')
      f.create_dataset('488 Channel', data = Images_488_Final, compression = 'gzip')



   

