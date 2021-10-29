#!/bin/bash
# Loop over each subject directory, place the processing script in the directory and execute it.
for directory in Data/SCSeg/* ; do
	echo "${directory}"
	cp Scripts/process_registration.sh "${directory}"/SC/.;
	cd "${directory}"/SC/; 
	# Run the processing script, suppressing stdout.
	./process_registration.sh 1> /dev/null;
	rm process_registration.sh; 
	cd ../../../..
done