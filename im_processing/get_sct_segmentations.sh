#!/bin/bash
# Loop over each subject directory, and get segmentations based on the pre-trained SCT models.
# These were not used for final results, but for a check when exploring the data.
# The final results are based on running the pre-trained models with our data pipeline, to ensure consistency across the results.
# SCT needs to be installed on the system to run.
cd ../data/SCSeg
for directory in ./* ; do
	echo "${directory}"
	cd "${directory}";
	# mkdir segmentation;
	# # Apply the SCT segmentation to the t2 and t2* images.
	sct_deepseg_lesion -i final/t2_iso_onT2srig_nl.nii.gz -c t2 -ofolder segmentation/ -r 1 -brain 0 -v 0
	sct_deepseg_lesion -i final/t2sMerge_iso.nii.gz 	 -c t2s -ofolder segmentation/ -r 1 -brain 0 -v 0
	# To test the SCT lesion segmentation on the original t2* images:
	# sct_deepseg_lesion -i SC/t2sMerge.nii.gz 	 -c t2s -ofolder segmentation/ -r 1 -brain 0 -v 0
	# Clean up temporary files from the segmentation.
	rm final/*RPI*
	cd ../;
done