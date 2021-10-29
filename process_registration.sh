mkdir res;

# Detect the centerline of the spinal cord.
sct_get_centerline -i t2sMerge.nii.gz -c t2s

# Create a mask around the previously detected centerline. 
# Set size slightly bigger than our eventual desired patch size to make sure we capture the relevant area in both T2s and later in T2.
sct_create_mask -size 30mm -i t2sMerge.nii.gz -p centerline,t2sMerge_centerline.nii.gz

# Crop the image around the centerline based on the mask just created.
sct_crop_image -i t2sMerge.nii.gz -m mask_t2sMerge.nii.gz -o t2sMerge_crop.nii.gz

# Resample the cropped t2s image to isotropic voxel dimensions of size 0.5mm
sct_resample -i t2sMerge_crop.nii.gz -mm 0.5x0.5x0.5 -o res/t2sMerge_iso.nii.gz

# Map the t2 image into cropped isotropic T2* image. This converts the T2 to isotropic and crops it to the same region.
animaApplyTransformSerie -i t2.nii.gz -g res/t2sMerge_iso.nii.gz -o res/t2_iso.nii.gz -t ~/mri/tools/identityTr.xml
# Do the same for the image with lesion masks (already in T2* space, but want to convert to isotropic and crop).
animaApplyTransformSerie -i labelLesion.nii.gz -g res/t2sMerge_iso.nii.gz -o res/labelLesion_iso.nii.gz -t ~/mri/tools/identityTr.xml

# Create a binary mask specifying where the intensities are zero or non-zero. This will be used to specify the region used to calculate the transformations below.
animaThrImage -i res/t2sMerge_iso.nii.gz -o res/t2sMerge_iso_mask.nii.gz -t 0.0

# Registration from T2 to T2* (rigid transformation)
# -M mask image for block generation  -r fixed image (i.e. reference image)   -m moving image   -o output image.
animaPyramidalBMRegistration -M res/t2sMerge_iso_mask.nii.gz -r res/t2sMerge_iso.nii.gz -m res/t2_iso.nii.gz -o res/t2_iso_onT2srig.nii.gz

# Use the output of the previous transformation and try to get a better result by using a dense non-linear transformation.
animaDenseSVFBMRegistration -m res/t2_iso_onT2srig.nii.gz -r res/t2sMerge_iso.nii.gz -M res/t2sMerge_iso_mask.nii.gz -o res/t2_iso_onT2srig_nl.nii.gz  --sp 2 -s 0.001 --opt 1 --sr 1 --fr 0.01 -a 0 -p 3 -l 1 --sym-reg 0 -T 3 --es 5
