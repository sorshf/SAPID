#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Soroush Shahryari Fard
# =============================================================================
"""This mudule uses Crux command line to run Percolator on the downsampled pep.xml files."""
# =============================================================================
# Imports
from Constants import comet_downsampled_output_dir, database_address, comet_pep_xml_dir
from Constants import num_of_DSratios
from Constants import comet_downsampled_output_dir
from Constants import decoy_prefix
from Constants import percolator_output_dir
import os


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

    file_names.sort()
    return file_names

def run_percolator(input_dir, input_name, output_dir, output_name, fasta_address):
    """This function runs Percolator on a pep.xml file and store it in a output directory

    Args:
        input_dir ([string]): Path to the folder containing the pep.xml files
        input_name ([string]): The name of the file with .pep.xml extension
        output_dir ([string]): Path to the output directory folder
        output_name ([string]): The name of the output for the Percolator's output
        fasta_address ([string]): Complete path and name to the fasta protein sequence database
    """
    print(decoy_prefix)
    print(output_name)
    print(output_dir)
    print(fasta_address)
    os.system(f'crux percolator --decoy-prefix {decoy_prefix} --decoy-xml-output F --fileroot {output_name} --output-dir "{output_dir}" --pepxml-output F --picked-protein "{fasta_address}" --pout-output F --protein-enzyme trypsin --search-input concatenated "{input_dir+input_name}"')

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

    #Get the paths
    input_dir = comet_downsampled_output_dir
    output_dir = percolator_output_dir
    fasta_address = database_address
    
    #Get the ORIGINAL COMET pep.xml file names
    file_names = (get_file_names(comet_pep_xml_dir))

    #Get the ratios for downsampling
    ratios = get_the_DSratios(num_of_DSratios)

    #Run Percolator on the pep.xml files
    for file_name in file_names:
        for ratio in ratios:
            # Note: we want all the LCM4 experiment files which is the testing experiment (not LCM3) 
            if "LCM3" not in file_name:
                input_name = file_name+"_"+ratio+".pep.xml"
                run_percolator(input_dir, input_name, output_dir, input_name.replace(".pep.xml",""), fasta_address)

if __name__=="__main__":
    main()
