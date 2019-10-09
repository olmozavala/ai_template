import os
from pandas import DataFrame
import pandas as pd
from inout.io_mri import *
from os.path import join
from config.MainConfig import *
from models.metrics import *
import visualization.utilsviz as utilviz
import SimpleITK as sitk

if __name__ == '__main__':

    root_folder = '/home/olmozavala/Desktop/TEST'
    ctrs=['Bowel','KidneyBilat','Skin','Duodenum','Liver',
          'SpinalCord','GTVp','Pancreas','Stomach']

    orig_img_name = 'img_tra.nrrd'

    orig_img = sitk.ReadImage(join(root_folder,orig_img_name))
    out_dims = orig_img.GetSize()
    print(F'Final dimentions are {out_dims}')

    utilviz.view_results = True
    all_ctrs_itk = []
    for c_ctr in ctrs:
        print(F'Working with contour {c_ctr}')
        itk_ctr = sitk.ReadImage(join(root_folder,F'ctr_{c_ctr}.nrrd'))
        all_ctrs_itk.append(itk_ctr)

    utilviz.drawMultipleSeriesItk([orig_img], contours=all_ctrs_itk, labels=ctrs)
