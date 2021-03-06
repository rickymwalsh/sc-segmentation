#!/usr/bin/env python

__author__ = 'Simon_2'

# ======================================================================================================================

# Extract results from .txt files generated by " " and draw box-and-whisker plots comparing the absolute error of
# estimation using the different metric extraction methods with different levels of noise and tracts std.

# ======================================================================================================================
import os
import glob
import getopt

import sys
import numpy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import *
# import subprocess
# path_sct = subprocess.check_output("echo %SCT_DIR%", shell=True)

class Param:
    def __init__(self):
        self.debug = 0
        self.results_folder = "results_20150210_200iter"
        self.methods_to_display = 'wath,ml,map'
        self.fname_folder_to_save_fig = './result_plots' #/Users/slevy_local/Dropbox/article_wm_atlas/fig/to_include_in_article'
        self.noise_std_to_display = 10
        self.tracts_std_to_display = 10
        self.csf_value_to_display = 5
        self.nb_RL_labels = 15

class Color:
    def __init__(self):
        self.purple = '\033[95m'
        self.cyan = '\033[96m'
        self.darkcyan = '\033[36m'
        self.blue = '\033[94m'
        self.green = '\033[92m'
        self.yellow = '\033[93m'
        self.red = '\033[91m'
        self.bold = '\033[1m'
        self.underline = '\033[4m'
        self.end = '\033[0m'

# =======================================================================================================================
# main
# =======================================================================================================================
def main():
    results_folder = param_default.results_folder
    methods_to_display = param_default.methods_to_display
    noise_std_to_display = param_default.noise_std_to_display
    tracts_std_to_display = param_default.tracts_std_to_display
    csf_value_to_display = param_default.csf_value_to_display
    nb_RL_labels = param_default.nb_RL_labels

    # Parameters for debug mode
    if param_default.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        results_folder = "/Users/slevy_local/spinalcordtoolbox/dev/atlas/validate_atlas/results_20150210_200iter"#"C:/cygwin64/home/Simon_2/data_methods_comparison"
        path_sct = '/Users/slevy_local/spinalcordtoolbox' #'C:/cygwin64/home/Simon_2/spinalcordtoolbox'
    else:
        path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))

        # Check input parameters
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'i:m:')  # define flags
        except getopt.GetoptError as err:  # check if the arguments are defined
            print str(err)  # error
            # usage() # display usage
        # if not opts:
        #     print 'Please enter the path to the result folder. Exit program.'
        #     sys.exit(1)
        #     # usage()
        for opt, arg in opts:  # explore flags
            if opt in '-i':
                results_folder = arg
            if opt in '-m':
                methods_to_display = arg

    # Append path that contains scripts, to be able to load modules
    sys.path.append(os.path.join(path_sct, "scripts"))
    import sct_utils as sct
    import isct_get_fractional_volume

    sct.printv("Working directory: " + os.getcwd())

    results_folder_noise = results_folder + '/noise'
    results_folder_tracts = results_folder + '/tracts'
    results_folder_csf = results_folder + '/csf'

    sct.printv('\n\nData will be extracted from folder ' + results_folder_noise, ' + results_folder_tracts + ' and ' + results_folder_csf, 'warning')
    sct.printv('\t\tCheck existence...')
    sct.check_folder_exist(results_folder_noise)
    sct.check_folder_exist(results_folder_tracts)
    sct.check_folder_exist(results_folder_csf)

    # Extract methods to display
    methods_to_display = methods_to_display.strip().split(',')

    # Extract file names of the results files
    fname_results_noise = glob.glob(os.path.join(results_folder_noise, "*.txt"))
    fname_results_tracts = glob.glob(os.path.join(results_folder_tracts, "*.txt"))
    fname_results_csf = glob.glob(os.path.join(results_folder_csf, "*.txt"))
    fname_results = fname_results_noise + fname_results_tracts + fname_results_csf
    # Remove doublons (due to the two folders)
    # for i_fname in range(0, len(fname_results)):
    #     for j_fname in range(0, len(fname_results)):
    #         if (i_fname != j_fname) & (os.path.basename(fname_results[i_fname]) == os.path.basename(fname_results[j_fname])):
    #             fname_results.remove(fname_results[j_fname])
    file_results = []
    for fname in fname_results:
        file_results.append(os.path.basename(fname))
    for file in file_results:
        if file_results.count(file) > 1:
            ind = file_results.index(file)
            fname_results.remove(fname_results[ind])
            file_results.remove(file)

    nb_results_file = len(fname_results)

    # 1st dim: SNR, 2nd dim: tract std, 3rd dim: mean abs error, 4th dim: std abs error
    # result_array = numpy.empty((nb_results_file, nb_results_file, 3), dtype=object)
    # SNR
    snr = numpy.zeros((nb_results_file))
    # Tracts std
    tracts_std = numpy.zeros((nb_results_file))
    # CSF value
    csf_values = numpy.zeros((nb_results_file))
    # methods' name
    methods_name = []  #numpy.empty((nb_results_file, nb_method), dtype=object)
    # labels
    error_per_label = []
    std_per_label = []
    labels_id = []
    # median
    median_results = numpy.zeros((nb_results_file, 5))
    # median std across bootstraps
    median_std = numpy.zeros((nb_results_file, 5))
    # min
    min_results = numpy.zeros((nb_results_file, 5))
    # max
    max_results = numpy.zeros((nb_results_file, 5))

    #
    for i_file in range(0, nb_results_file):

        # Open file
        f = open(fname_results[i_file])  # open file
        # Extract all lines in .txt file
        lines = [line for line in f.readlines() if line.strip()]

        # extract SNR
        # find all index of lines containing the string "sigma noise"
        ind_line_noise = [lines.index(line_noise) for line_noise in lines if "sigma noise" in line_noise]
        if len(ind_line_noise) != 1:
            sct.printv("ERROR: number of lines including \"sigma noise\" is different from 1. Exit program.", 'error')
            sys.exit(1)
        else:
            # result_array[:, i_file, i_file] = int(''.join(c for c in lines[ind_line_noise[0]] if c.isdigit()))
            snr[i_file] = int(''.join(c for c in lines[ind_line_noise[0]] if c.isdigit()))

        # extract tract std
        ind_line_tract_std = [lines.index(line_tract_std) for line_tract_std in lines if
                              "range tracts" in line_tract_std]
        if len(ind_line_tract_std) != 1:
            sct.printv("ERROR: number of lines including \"range tracts\" is different from 1. Exit program.", 'error')
            sys.exit(1)
        else:
            # result_array[i_file, i_file, :] = int(''.join(c for c in lines[ind_line_tract_std[0]].split(':')[1] if c.isdigit()))
            # regex = re.compile(''('(.*)':)  # re.I permet d'ignorer la case (majuscule/minuscule)
            # match = regex.search(lines[ind_line_tract_std[0]])
            # result_array[:, i_file, :, :] = match.group(1)  # le groupe 1 correspond a '.*'
            tracts_std[i_file] = int(''.join(c for c in lines[ind_line_tract_std[0]].split(':')[1] if c.isdigit()))

        # extract CSF value
        ind_line_csf_value = [lines.index(line_csf_value) for line_csf_value in lines if
                              "# value CSF" in line_csf_value]
        if len(ind_line_csf_value) != 1:
            sct.printv("ERROR: number of lines including \"range tracts\" is different from 1. Exit program.", 'error')
            sys.exit(1)
        else:
            # result_array[i_file, i_file, :] = int(''.join(c for c in lines[ind_line_tract_std[0]].split(':')[1] if c.isdigit()))
            # regex = re.compile(''('(.*)':)  # re.I permet d'ignorer la case (majuscule/minuscule)
            # match = regex.search(lines[ind_line_tract_std[0]])
            # result_array[:, i_file, :, :] = match.group(1)  # le groupe 1 correspond a '.*'
            csf_values[i_file] = int(''.join(c for c in lines[ind_line_csf_value[0]].split(':')[1] if c.isdigit()))


        # extract method name
        ind_line_label = [lines.index(line_label) for line_label in lines if "Label" in line_label]
        if len(ind_line_label) != 1:
            sct.printv("ERROR: number of lines including \"Label\" is different from 1. Exit program.", 'error')
            sys.exit(1)
        else:
            # methods_name[i_file, :] = numpy.array(lines[ind_line_label[0]].strip().split(',')[1:])
            methods_name.append(lines[ind_line_label[0]].strip().replace(' ', '').split(',')[1:])

        # extract median
        ind_line_median = [lines.index(line_median) for line_median in lines if "median" in line_median]
        if len(ind_line_median) != 1:
            sct.printv("WARNING: number of lines including \"median\" is different from 1. Exit program.", 'warning')
            # sys.exit(1)
        else:
            median = lines[ind_line_median[0]].strip().split(',')[1:]
            # result_array[i_file, i_file, 0] = [float(m.split('(')[0]) for m in median]
            median_results[i_file, :] = numpy.array([float(m.split('(')[0]) for m in median])
            median_std[i_file, :] = numpy.array([float(m.split('(')[1][:-1]) for m in median])

        # extract min
        ind_line_min = [lines.index(line_min) for line_min in lines if "min," in line_min]
        if len(ind_line_min) != 1:
            sct.printv("WARNING: number of lines including \"min\" is different from 1. Exit program.", 'warning')
            # sys.exit(1)
        else:
            min = lines[ind_line_min[0]].strip().split(',')[1:]
            # result_array[i_file, i_file, 1] = [float(m.split('(')[0]) for m in min]
            min_results[i_file, :] = numpy.array([float(m.split('(')[0]) for m in min])

        # extract max
        ind_line_max = [lines.index(line_max) for line_max in lines if "max" in line_max]
        if len(ind_line_max) != 1:
            sct.printv("WARNING: number of lines including \"max\" is different from 1. Exit program.", 'warning')
            # sys.exit(1)
        else:
            max = lines[ind_line_max[0]].strip().split(',')[1:]
            # result_array[i_file, i_file, 1] = [float(m.split('(')[0]) for m in max]
            max_results[i_file, :] = numpy.array([float(m.split('(')[0]) for m in max])

        # extract error for each label
        error_per_label_for_file_i = []
        std_per_label_for_file_i = []
        labels_id_for_file_i = []
        # Due to 2 different kind of file structure, the number of the last label line must be adapted
        if not ind_line_median:
            ind_line_median = [len(lines) + 1]
        for i_line in range(ind_line_label[0] + 1, ind_line_median[0] - 1):
            line_label_i = lines[i_line].strip().split(',')
            error_per_label_for_file_i.append([float(error.strip().split('(')[0]) for error in line_label_i[1:]])
            std_per_label_for_file_i.append([float(error.strip().split('(')[1][:-1]) for error in line_label_i[1:]])
            labels_id_for_file_i.append(int(line_label_i[0]))
        error_per_label.append(error_per_label_for_file_i)
        std_per_label.append(std_per_label_for_file_i)
        labels_id.append(labels_id_for_file_i)

        # close file
        f.close()

    # check if all the files in the result folder were generated with the same number of methods
    if not all(x == methods_name[0] for x in methods_name):
        sct.printv(
            'ERROR: All the generated files in folder ' + results_folder + ' have not been generated with the same number of methods. Exit program.',
            'error')
        sys.exit(1)
    # check if all the files in the result folder were generated with the same labels
    if not all(x == labels_id[0] for x in labels_id):
        sct.printv(
            'ERROR: All the generated files in folder ' + results_folder + ' have not been generated with the same labels. Exit program.',
            'error')
        sys.exit(1)

    # convert the list "error_per_label" into a numpy array to ease further manipulations
    error_per_label = numpy.array(error_per_label)
    std_per_label = numpy.array(std_per_label)
    # compute different stats
    abs_error_per_labels = numpy.absolute(error_per_label)
    max_abs_error_per_meth = numpy.amax(abs_error_per_labels, axis=1)
    min_abs_error_per_meth = numpy.amin(abs_error_per_labels, axis=1)
    mean_abs_error_per_meth = numpy.mean(abs_error_per_labels, axis=1)
    std_abs_error_per_meth = numpy.std(abs_error_per_labels, axis=1)

    # average error and std across sides
    meanRL_abs_error_per_labels = numpy.zeros((error_per_label.shape[0], nb_RL_labels, error_per_label.shape[2]))
    meanRL_std_abs_error_per_labels = numpy.zeros((std_per_label.shape[0], nb_RL_labels, std_per_label.shape[2]))
    for i_file in range(0, nb_results_file):
        for i_meth in range(0, len(methods_name[i_file])):
            for i_label in range(0, nb_RL_labels):
                # find indexes of corresponding labels
                ind_ID_first_side = labels_id[i_file].index(i_label)
                ind_ID_other_side = labels_id[i_file].index(i_label + nb_RL_labels)
                # compute mean across 2 sides
                meanRL_abs_error_per_labels[i_file, i_label, i_meth] = float(error_per_label[i_file, ind_ID_first_side, i_meth] + error_per_label[i_file, ind_ID_other_side, i_meth]) / 2
                meanRL_std_abs_error_per_labels[i_file, i_label, i_meth] = float(std_per_label[i_file, ind_ID_first_side, i_meth] + std_per_label[i_file, ind_ID_other_side, i_meth]) / 2

    nb_method = len(methods_to_display)

    sct.printv('Noise std of the ' + str(nb_results_file) + ' generated files:')
    print snr
    print '----------------------------------------------------------------------------------------------------------------'
    sct.printv('Tracts std of the ' + str(nb_results_file) + ' generated files:')
    print tracts_std
    print '----------------------------------------------------------------------------------------------------------------'
    sct.printv('CSF value of the ' + str(nb_results_file) + ' generated files:')
    print csf_values
    print '----------------------------------------------------------------------------------------------------------------'
    sct.printv('Methods used to generate results for the ' + str(nb_results_file) + ' generated files:')
    print methods_name
    print '----------------------------------------------------------------------------------------------------------------'
    sct.printv('Median obtained with each method (in colons) for the ' + str(nb_results_file) + ' generated files (in lines):')
    print median_results
    print '----------------------------------------------------------------------------------------------------------------'
    sct.printv('Minimum obtained with each method (in colons) for the ' + str(
        nb_results_file) + ' generated files (in lines):')
    print min_results
    print '----------------------------------------------------------------------------------------------------------------'
    sct.printv('Maximum obtained with each method (in colons) for the ' + str(
        nb_results_file) + ' generated files (in lines):')
    print max_results
    print '----------------------------------------------------------------------------------------------------------------'
    sct.printv('Labels\' ID (in colons) for the ' + str(nb_results_file) + ' generated files (in lines):')
    print labels_id
    print '----------------------------------------------------------------------------------------------------------------'
    sct.printv('Errors obtained with each method (in colons) for the ' + str(nb_results_file) + ' generated files (in lines):')
    print error_per_label
    print '----------------------------------------------------------------------------------------------------------------'
    sct.printv('Mean errors across both sides obtained with each method (in colons) for the ' + str(nb_results_file) + ' generated files (in lines):')
    print meanRL_abs_error_per_labels


    # Compute fractional volume per label
    labels_id_FV, labels_name_FV, fract_vol_per_lab, labels_name_FV_RL_gathered, fract_vol_per_lab_RL_gathered = isct_get_fractional_volume.get_fractional_volume_per_label('./cropped_atlas/', 'info_label.txt')
    # # Get the number of voxels including at least one tract
    # nb_voxels_in_WM = isct_get_fractional_volume.get_nb_voxel_in_WM('./cropped_atlas/', 'info_label.txt')
    # normalize by the number of voxels in WM and express it as a percentage
    fract_vol_norm = numpy.divide(fract_vol_per_lab_RL_gathered, numpy.sum(fract_vol_per_lab_RL_gathered)/100)

    # NOT NECESSARY NOW WE AVERAGE ACROSS BOTH SIDES (which orders the labels)
    # # check if the order of the labels returned by the function computing the fractional volumes is the same (which should be the case)
    # if labels_id_FV != labels_id[0]:
    #     sct.printv('\n\nERROR: the labels IDs returned by the function \'i_sct_get_fractional_volume\' are different from the labels IDs of the results files\n\n', 'error')

    # # Remove labels #30 and #31
    # labels_id_FV_29, labels_name_FV_29, fract_vol_per_lab_29 = labels_id_FV[:-2], labels_name_FV[:-2], fract_vol_per_lab[:-2]

    # indexes of labels sort according to the fractional volume
    ind_labels_sort = numpy.argsort(fract_vol_norm)

    # Find index of the file generated with noise variance = 10 and tracts std = 10
    ind_file_to_display = numpy.where((snr == noise_std_to_display) & (tracts_std == tracts_std_to_display) & (csf_values == csf_value_to_display))

    # sort arrays in this order
    meanRL_abs_error_per_labels_sort = meanRL_abs_error_per_labels[ind_file_to_display[0], ind_labels_sort, :]
    meanRL_std_abs_error_per_labels_sort = meanRL_std_abs_error_per_labels[ind_file_to_display[0], ind_labels_sort, :]
    labels_name_sort = numpy.array(labels_name_FV_RL_gathered)[ind_labels_sort]

    # *********************************************** START PLOTTING HERE **********************************************

    # stringColor = Color()
    matplotlib.rcParams.update({'font.size': 50, 'font.family': 'trebuchet'})
    # plt.rcParams['xtick.major.pad'] = '11'
    plt.rcParams['ytick.major.pad'] = '15'

    fig = plt.figure(figsize=(60, 37))
    width = 1.0 / (nb_method + 1)
    ind_fig = numpy.arange(len(labels_name_sort)) * (1.0 + width)
    plt.ylabel('Absolute error (%)\n', fontsize=65)
    plt.xlabel('Fractional volume (% of the total number of voxels in WM)', fontsize=65)
    plt.title('Absolute error per tract as a function of their fractional volume\n\n', fontsize=30)
    plt.suptitle('(Noise std='+str(snr[ind_file_to_display[0]][0])+', Tracts std='+str(tracts_std[ind_file_to_display[0]][0])+', CSF value='+str(csf_values[ind_file_to_display[0]][0])+')', fontsize=30)

    # colors = plt.get_cmap('jet')(np.linspace(0, 1.0, nb_method))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['o', 's', '^', 'D']
    errorbar_plots = []
    for meth, color, marker in zip(methods_to_display, colors, markers):
        i_meth = methods_name[0].index(meth)
        i_meth_to_display = methods_to_display.index(meth)

        plot_i = plt.errorbar(ind_fig + i_meth_to_display * width, meanRL_abs_error_per_labels_sort[:, i_meth], meanRL_std_abs_error_per_labels_sort[:, i_meth], color=color, marker=marker, markersize=35, lw=7, elinewidth=1, capthick=5, capsize=10)
        # plot_i = plt.boxplot(numpy.transpose(abs_error_per_labels[ind_files_csf_sort, :, i_meth]), positions=ind_fig + i_meth_to_display * width + (float(i_meth_to_display) * width) / (nb_method + 1), widths=width, boxprops=boxprops, medianprops=medianprops, flierprops=flierprops, whiskerprops=whiskerprops, capprops=capprops)
        errorbar_plots.append(plot_i)

    # add alternated vertical background colored bars
    for i_xtick in range(0, len(ind_fig), 2):
        plt.axvspan(ind_fig[i_xtick] - width - width / 2, ind_fig[i_xtick] + (nb_method + 1) * width - width / 2, facecolor='grey', alpha=0.1)

    # concatenate value of fractional volume to labels'name
    xtick_labels = [labels_name_sort[i_lab]+'\n'+r'$\bf{['+str(round(fract_vol_norm[ind_labels_sort][i_lab], 2))+']}$' for i_lab in range(0, len(labels_name_sort))]
    ind_lemniscus = numpy.where(labels_name_sort == 'spinal lemniscus (spinothalamic and spinoreticular tracts)')[0][0]
    xtick_labels[ind_lemniscus] = 'spinal lemniscus\n'+r'$\bf{['+str(round(fract_vol_norm[ind_labels_sort][ind_lemniscus], 2))+']}$'

    # plt.legend(box_plots, methods_to_display, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    plt.legend(errorbar_plots, methods_to_display, loc=1, fontsize=50, numpoints=1)
    plt.xticks(ind_fig + (numpy.floor(float(nb_method-1)/2)) * width, xtick_labels, fontsize=45)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0, top=0.95, right=0.96)
    plt.gca().set_xlim([-width, numpy.max(ind_fig) + (nb_method + 0.5) * width])
    plt.gca().set_ylim([0, 17])
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1.0))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    plt.grid(b=True, axis='y', which='both')
    fig.autofmt_xdate()

    plt.savefig(os.path.join(param_default.fname_folder_to_save_fig, 'absolute_error_vs_fractional_volume.pdf'), format='PDF')

    plt.show(block=False)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    param_default = Param()
    # call main function
    main()
