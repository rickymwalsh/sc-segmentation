mkdir res;

sct_get_centerline -i t2sMerge.nii.gz -c t2s

sct_create_mask -size 24mm -i t2sMerge.nii.gz -p centerline,t2sMerge_centerline.nii.gz

sct_crop_image -i t2sMerge.nii.gz -m mask_t2sMerge.nii.gz -o t2sMerge_crop.nii.gz

# imgCreateIsotropicVol.py t2sMerge_crop.nii.gz res/t2sMerge_iso.nii.gz
sct_resample -i t2sMerge.nii.gz -mm 0.5x0.5x0.5 -o res/t2sMerge_iso.nii.gz
sct_resample -i t2.nii.gz -mm 0.5x0.5x0.5 -o res/t2_iso.nii.gz

# May need to run this to get the necessary input for the following command.
# animaTransformSerieXmlGenerator -i transform_aff.txt -i transform_nl.nii.gz -o transforms.xml

animaApplyTransformSerie -i t2.nii.gz -g res/t2sMerge_iso.nii.gz -o res/t2_iso.nii.gz -t ~/WORK/SEP/Tools/identity.xml

signalMaskFromConditions.R res/t2_iso.nii.gz --exp 'data1!=0' -o res/t2_iso_masked.nii.gz --label 1 

animaPyramidalBMRegistration -M res/t2_iso_masked.nii.gz -r res/t2_iso.nii.gz -m res/t2sMerge_iso.nii.gz -o res/t2sMerge_iso_onT2rig.nii.gz

animaDenseSVFBMRegistration -m res/t2sMerge_iso_onT2rig.nii.gz -r res/t2_iso.nii.gz -M res/t2_iso_masked.nii.gz -o res/t2sMerge_iso_onT2rig_nl.nii.gz  --sp 2 -s 0.001 --opt 1 --sr 1 --fr 0.01 -a 0 -p 3 -l 1 --sym-reg 0 -T 3 --es 5

