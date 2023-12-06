#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Soroush Shahryari Fard
# =============================================================================
"""Convert each comet pep.xml into mzID and store them in their respective folder.
   Then, it runs ProteinProphet on each of the mzID files."""
# =============================================================================
# Imports
from Constants import num_of_DSratios
from Constants import comet_downsampled_output_dir
from Constants import PP_xml_mzid_dir
from Constants import comet_pep_xml_dir
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


def create_subdirs(maindirectory, DSratios):
    """This function creates a folder inside the main directory (if already doesn't exist)

    Args:
        maindirectory ([string]): Path to the directory where the folders are being created
        DSratios ([string]): The downsampling ratio value as string (e.g. "0.20")
    """
    for ratio in DSratios:
        #If the directory doesn't already exist
        if (not (os.access(str(maindirectory+"DSratio_"+ratio),os.F_OK))):
            os.mkdir(str(maindirectory+"DSratio_"+ratio)) #This line makes the directory


def get_the_DSratios(num_of_threshold):
    """Generate the downsampling ratio values

    Args:
        num_of_DSratios ([int]): Number of downsampling ratio needs to be generated

    Returns:
        [list]: List containing the downsampling ratios
    """
    #ratios = numpy.linspace(0,0.99,num_of_threshold)
    #return ["{:.2f}".format(round(thr, 3)) for thr in ratios]
    #return ["0.08", "0.17", "0.33", "0.41", "0.57", "0.65","0.82","0.90"]
    return ["0.00","0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90"]


def convert_pepXML_to_mzid(comet_pep_xml_dir, pepXML_name, output_mzid_dir):
    """This function converts pep.xml files into mzid files

    Args:
        comet_pep_xml_dir ([string]): Path where downsampled Comet pep.xml are present
        pepXML_name ([string]): The name of the pep.xml file 
        output_mzid_dir ([string]): Path where the mzid files need to be stored
    """
    os.system(str("idconvert " + comet_pep_xml_dir +  pepXML_name + " -o " + output_mzid_dir)) 


def runProphet(comet_pep_xml_dir, pepXML_name, output_prophet_dir, ratio):
    """This function runs ProteinProphet on the Comet pep.xml files

    Args:
        comet_pep_xml_dir ([string]): Path where downsampled Comet pep.xml are present
        pepXML_name ([string]): Name of the pep.xml file that needs to be evaluated by PP
        output_prophet_dir ([string]): Path where the output of the ProteinProphet needs to be stored
        ratio ([string]): The downsampling ratio value
    """
    os.system(str("xinteract " + "-N"+ output_prophet_dir+pepXML_name+".pep.xml" + " -p0.0 " + "-l7" + " -PPM" + " -Op " + comet_pep_xml_dir+pepXML_name+"_"+ratio+".pep.xml" ))

def main():
    #Get the ORIGINAL COMET pep.xml file names
    file_names = (get_file_names(comet_pep_xml_dir))

    #Get the ratios for downsampling
    ratios = get_the_DSratios(num_of_DSratios)
    
    #Create the subdirectories to store the mzid and pepxml
    create_subdirs(PP_xml_mzid_dir, ratios)

    #Convert each comet pep.xml into mzid and store them in their respective folder
    for file_name in file_names:
        for ratio in ratios:
            print((comet_downsampled_output_dir, file_name+"_"+ratio+".pep.xml", PP_xml_mzid_dir+"DSratio_"+ratio))
            convert_pepXML_to_mzid(comet_downsampled_output_dir, file_name+"_"+ratio+".pep.xml", PP_xml_mzid_dir+"DSratio_"+ratio)

    #Run TPP (xInteract on the downsampled file and store them in their respective folder)
    for file_name in file_names:
        for ratio in ratios:
            runProphet(comet_downsampled_output_dir, file_name, PP_xml_mzid_dir+"DSratio_"+ratio+"/", ratio)


if __name__ == "__main__":
    main()