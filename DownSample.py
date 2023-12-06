#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Soroush Shahryari Fard
# =============================================================================
"""This module downsamples the pep.xml files from Comet."""
# =============================================================================
# Imports
from Constants import comet_pep_xml_dir
from Constants import num_of_DSratios
from Constants import comet_downsampled_output_dir
import random 
import math
import os
import time


    
    
def get_file_names(directory):
    """Returns the .pep.xml file names for an experiment inside a directory

    Args:
        directory ([string]): [description]

    Returns:
        [list]: Output list of names. Each name is without the format extension.
    """
    file_names = []

    #Add the name of the files from "experiment" into the list
    for filename in os.listdir(directory):
        file_names.append(filename[:-8])

    #Sort the files
    file_names.sort()

    #Return the file names in a list
    return file_names


def downsample_CometPepXML(input_dir, input_name, output_dir, output_name, seed, downsample_ratio=1):
    """Function to downsample Comet's pep.xml file

    Args:
        input_dir ([string]): path to directory containing the pep.xml files
        input_name ([string]): name of the pep.xml file which needs to be downsampled
        output_dir ([string]): path to directory where the downsampled files need to be stored
        output_name (string): name of the downsampled pep.xml file
        seed ([string]): a seed for the random downsampling of the spectra
        downsample_ratio (int, optional): The ratio of downsampling. Defaults to 1.

    Returns:
        [void]: writes the output file in the putput directory
    """
    #Function that returns the indexs of a pattern in a string
    def findall(p, s):
        to_return = []
        i = s.find(p)
        while i != -1:
            to_return.append(i)
            i = s.find(p, i+1)
        return to_return
    
    #Read the Comet file
    with open(input_dir+input_name, "r", encoding='utf-8') as f:
        string=f.read()
    
    #Read the head and footter
    heading = string[0:string.find("<spectrum_query")]
    footer =  "\n</msms_run_summary>\n </msms_pipeline_analysis>"
    
    #Get the indexes
    start_indexes = findall("<spectrum_query", string)
    closing_indexes = findall("</spectrum_query>", string)
    
    #Read each spectrum and put them in a list
    spectrum_list = []
    for start_ind, end_ind in zip(start_indexes,closing_indexes):
        spectrum_list.append(string[start_ind: end_ind+17])
        
    #Write the output file
    random.seed(hash(seed))
    with open(output_dir+output_name, mode='w',encoding='utf-8',newline=None) as f:
        f.write(heading)
        for i in random.sample(spectrum_list, k=math.floor(len(spectrum_list)*(1-downsample_ratio))):
            f.write(i)
        f.write(footer)

def get_the_DSratios(num_of_DSratios):
    """Generate the downsampling ratio values

    Args:
        num_of_DSratios ([int]): Number of downsampling ratio needs to be generated

    Returns:
        [list]: List containing the downsampling ratios
    """
    #In case you want different downsampling ratios, use the following two lines
    #ratios = numpy.linspace(0,0.99,num_of_DSratios)
    #return ["{:.2f}".format(round(thr, 3)) for thr in ratios]

    #We only want the following downsampling ratios
    return ["0.00","0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90"]
    
def main():
    
    #Get the ratios for downsampling
    ratios = get_the_DSratios(num_of_DSratios)

    #Get the file name
    file_names = get_file_names(comet_pep_xml_dir)

    #Make the downsampled Comet output files
    for ratio in ratios:
        for file_name in file_names:

            #This is just to keep track of the time
            time1 = time.time()

            #Downsample the files
            downsample_CometPepXML(comet_pep_xml_dir, file_name+".pep.xml", comet_downsampled_output_dir,
             file_name+"_"+ratio+".pep.xml", file_name+"_"+ratio+".pep.xml", downsample_ratio=float(ratio))

            #print the time it took for a file to be downsampled
            print(file_name, "took", time.time()-time1, "to downsample", ratio)


if __name__=="__main__":
    main()