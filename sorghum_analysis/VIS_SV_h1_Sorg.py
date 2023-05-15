from plantcv import plantcv as pcv
import matplotlib
import cv2
import numpy as np
import argparse
from  matplotlib import pyplot as plt
import os
from plantcv.parallel import workflow_inputs
import datetime

start = datetime.datetime.now()

# Set input variables
args = workflow_inputs() 

print(args.image1)    
# Set variables
pcv.params.debug = args.debug     # Replace the hard-coded debug with the debug flag

#use image1 because of the new workflow inputs
img, path, filename = pcv.readimage(filename=args.image1)
filename = os.path.split(args.image1)[1]
    
#define new color correct function for logv images
def affine_color_correction(img, source_matrix, target_matrix):
    h,w,c = img.shape
    n = source_matrix.shape[0]
    S = np.concatenate((source_matrix[:,1:].copy(),np.ones((n,1))),axis=1)
    T = target_matrix[:,1:].copy()
    
    tr = T[:,0]
    tg = T[:,1]
    tb = T[:,2]
    
    ar = np.matmul(np.linalg.pinv(S), tr)
    ag = np.matmul(np.linalg.pinv(S), tg)
    ab = np.matmul(np.linalg.pinv(S), tb)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pix = np.concatenate((img_rgb.reshape(h*w,c).astype(np.float64)/255, np.ones((h*w,1))), axis=1)
    
    img_r_cc = (255*np.clip(np.matmul(img_pix,ar),0,1)).astype(np.uint8)
    img_g_cc = (255*np.clip(np.matmul(img_pix,ag),0,1)).astype(np.uint8)
    img_b_cc = (255*np.clip(np.matmul(img_pix,ab),0,1)).astype(np.uint8)
    
    img_cc = np.stack((img_b_cc,img_g_cc,img_r_cc), axis=1).reshape(h,w,c)
    
    return img_cc
    
#log correct the plant image  #for 2023 cut out everything to right of '=' and replace it with 'img' so plact_logv = img 
plant_logv = img    
#color correct the logv image
dataframe2, start2, space2 = pcv.transform.find_color_card(rgb_img=plant_logv, background='light')

source_mask = pcv.transform.create_color_card_mask(plant_logv, radius=10, start_coord=start2, 
                                                   spacing=space2, nrows=4, ncols=6)
headers, source_matrix = pcv.transform.get_color_matrix(rgb_img=plant_logv, mask=source_mask)

#load the color card values that they should be
target_matrix = pcv.transform.load_matrix(filename='/shares/nshakoor_share/users/jstanton/phenotyper_data/2022/x-rite_color_matrix_k2.npz')

color_corrected_img = affine_color_correction(plant_logv, source_matrix, target_matrix)

#mask on a/b for threshold 1
thresh1 = pcv.threshold.dual_channels(rgb_img = color_corrected_img, x_channel = "a", y_channel = "b", points = [(100,130),(130,175)], above=True, max_value=255)

#get rid of noise
thresh1_fill = pcv.fill(bin_img=thresh1, size=3)
    
#threshold using Naive Bayes
thresh2 = pcv.naive_bayes_classifier(rgb_img=color_corrected_img, 
                                  pdf_file="/shares/nshakoor_share/users/jstanton/phenotyper_data/2022/maize_naive_bayes_pdfs_new.txt")

#get green and purple plants from naive bayes
thresh2_plant = pcv.logical_or(thresh2['greenplant'], thresh2['purpleplant'])

#get rid of noise in the naive bayes threshold
thresh2_fill = pcv.fill(bin_img=thresh2_plant, size=100)

#use an ROI to get the naive bayes mask only on the plant (it's ok if it misses some, a/b will get it
id_objects_nb, obj_hierarchy_nb = pcv.find_objects(img=color_corrected_img, mask=thresh2_fill)
roi_nb, roi_hierarchy_nb= pcv.roi.rectangle(img=color_corrected_img, x=1250, y=700, h=1275, w=1700)
roi_objects_nb, hierarchy_nb, kept_mask_nb, obj_area_nb = pcv.roi_objects(img=color_corrected_img, roi_contour=roi_nb, 
                                                               roi_hierarchy=roi_hierarchy_nb, 
                                                              object_contour=id_objects_nb, 
                                                               obj_hierarchy=obj_hierarchy_nb,
                                                               roi_type='cutto')
#combine the a/b and naive bayes masks
mask_final_combo = pcv.logical_or(bin_img1=thresh1_fill, bin_img2=kept_mask_nb)

# Fill in small objects to make sure a/b and naive bayes combined well
mask_final_filled_holes = pcv.closing(gray_img=mask_final_combo)

#find objects
id_objects_abnb, obj_hierarchy_abnb = pcv.find_objects(img=color_corrected_img, mask=mask_final_filled_holes)

#define an ROI
roi_abnb, roi_hierarchy_abnb= pcv.roi.rectangle(img=color_corrected_img, x=1250, y=700, h=1275, w=1700)

roi_objects_abnb, hierarchy_abnb, kept_mask_abnb, obj_area_abnb = pcv.roi_objects(img=color_corrected_img, roi_contour=roi_abnb, 
                                                               roi_hierarchy=roi_hierarchy_abnb, 
                                                               object_contour=id_objects_abnb, 
                                                               obj_hierarchy=obj_hierarchy_abnb,
                                                               roi_type='partial')

if obj_area_abnb > 4:
    #combine kept objects
    obj_combined_abnb, mask_final_combo_abnb = pcv.object_composition(img=color_corrected_img, contours=roi_objects_abnb, hierarchy=hierarchy_abnb)

    # Find shape properties, data gets stored to an Outputs class automatically
    analysis_image = pcv.analyze_object(img=color_corrected_img, obj=obj_combined_abnb, mask=mask_final_combo_abnb, label="default")
    boundary_image = pcv.analyze_bound_horizontal(img=color_corrected_img, obj=obj_combined_abnb, mask=mask_final_combo_abnb, 
                                               line_position=1980, label="default")
    # Determine color properties
    color_histogram = pcv.analyze_color(rgb_img=color_corrected_img, mask=mask_final_combo_abnb, colorspaces='all', label="default")
    
    # Print the image out to save it 
    pcv.print_image(analysis_image, os.path.join(args.outdir, filename + "_shapes.jpg"))
    
    # Write shape and color data to results file
    pcv.outputs.save_results(filename=args.result)

end = datetime.datetime.now()
duration = end - start
print(f"{args.image1}: {duration.seconds}")
