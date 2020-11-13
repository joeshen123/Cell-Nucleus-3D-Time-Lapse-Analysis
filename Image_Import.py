from nd2reader.reader import ND2Reader
import numpy as np
import time
from tkinter import ttk
from tkinter import simpledialog
from tkinter import filedialog
from tkinter import messagebox
import tkinter as tk
import matplotlib.pyplot as plt
from skimage import io
import cv2
from skimage import img_as_ubyte
import warnings
import h5py
from colorama import Fore
from tqdm import tqdm

#Ignore warnings issued by skimage through conversion to uint8
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)

# Use tkinter to interactively select files to import
root = tk.Tk()
root.withdraw()

my_filetypes = [('all files', '.*'),('Movie files', '.nd2')]

Image_Stack_Path = filedialog.askopenfilename(title='Please Select a Movie', filetypes = my_filetypes)


# Define a function to convert time series of ND2 images to a numpy list of 
# images (t,z,y,x).

answer = messagebox.askyesno("Question","Does the image contain multiple z stacks?")



def Z_Stack_Images_Extractor(address, fields_of_view, z_answer):
   Image_Sequence = ND2Reader(address)
   Channel_list = Image_Sequence.metadata['channels']
   
   # Check whether the data has one channel or two channels. If only one channel, put GUV_Channel and Protein_Channel = 0
   if len(Channel_list) == 1:
      GUV_Channel = 0
      Protein_Channel = 0

   # Select correct channels for downstream import
   elif 'DsRed' in Channel_list[0] or '561' in Channel_list[0]:
      GUV_Channel = 0
      Protein_Channel = 1

   else:
      GUV_Channel = 1
      Protein_Channel = 0

   time_series = Image_Sequence.sizes['t']
   
   if z_answer:
     z_stack = Image_Sequence.sizes['z']
   
   Intensity_Slice = []
   GUV_Slice = []

   n = 0

   # create progress bar
   pb = tqdm(range(time_series), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET))

   for time in pb:
     
     pb.set_description("Converting images to numpy arrays")

     if z_answer:
       z_stack_images = [] 
       z_stack_Intensity_images = []
       for z_slice in range(z_stack):
          slice = Image_Sequence.get_frame_2D(c=GUV_Channel, t=time, z=z_slice, v=fields_of_view)
          Intensity_slice = Image_Sequence.get_frame_2D(c=Protein_Channel, t=time, z=z_slice, v=fields_of_view)
          z_stack_images.append(slice)
          z_stack_Intensity_images.append(Intensity_slice)
     


       z_stack_images = np.array(z_stack_images)
       z_stack_Intensity_images = np.array(z_stack_Intensity_images)
     
     else:
        z_stack_images = Image_Sequence.get_frame_2D(c=GUV_Channel, t=time, v=fields_of_view)
        z_stack_Intensity_images = Image_Sequence.get_frame_2D(c=Protein_Channel, t=time, v=fields_of_view)

     GUV_Slice.append(z_stack_images)

     Intensity_Slice.append(z_stack_Intensity_images)
     
   GUV_Slice = np.array(GUV_Slice)
   Intensity_Slice = np.array(Intensity_Slice)



   return (GUV_Slice, Intensity_Slice)


Image_Sequence = ND2Reader(Image_Stack_Path)
FOV_list = Image_Sequence.metadata['fields_of_view']

GUV_Image_list = []
Intensity_list = []

for fov in range(len(FOV_list)):
   GUV_Images, Image_Intensity = Z_Stack_Images_Extractor(Image_Stack_Path,fields_of_view=fov,z_answer=answer)
   GUV_Image_list.append(GUV_Images)
   Intensity_list.append(Image_Intensity)


File_save_names = '.'.join(Image_Stack_Path.split(".")[:-1])

for n in range(len(FOV_list)):
   GUV_Image_Name='{File_Name}_{num}.hdf5'.format(File_Name = File_save_names, num = n + 1)
   
   GUV_Images = GUV_Image_list[n]
   Image_Intensity = Intensity_list[n]

   with h5py.File(GUV_Image_Name, "w") as f:
      f.create_dataset('488 Channel', data = Image_Intensity, compression = 'gzip')
      f.create_dataset('561 Channel', data = GUV_Images, compression = 'gzip')



   

