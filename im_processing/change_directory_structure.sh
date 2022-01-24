#!/bin/bash
# Loop over each subject directory, and tidy up the directories.
# This was needed since all of the processing had been done before deciding on the optimal directory structure, 
# but the updated processing code should take care of this from the start.
cd ../data/SCSeg
for directory in ./* ; do
	echo "${directory}"
	cd "${directory}";
	mkdir intermediate patches final;
	mv SC/res/patches* patches/;
	mv SC/res/t2_iso_onT2srig_nl.nii.gz SC/res/t2sMerge_iso.nii.gz SC/res/labelLesion_iso_bin.nii.gz final/;
	mv SC/res/* intermediate/;
	rm -r SC/res/;
	mv SC/* intermediate/;
	mv intermediate/t2.nii.gz intermediate/t2sMerge.nii.gz intermediate/labelLesion.nii.gz SC/;
	cd ../;
done