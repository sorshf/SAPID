#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Soroush Shahryari Fard
# =============================================================================
"""Feature engineering and tabular dataset creation happens here."""
# =============================================================================
# Imports
import pyteomics.mzid, pyteomics.protxml
import pandas as pd
import os
import re as regex
import numpy as np
import time
import codecs
import statistics
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import itertools
from Constants import decoy_prefix
from Constants import PP_xml_mzid_dir
from Constants import output_spatial_data_dir
from Constants import num_of_DSratios



def get_file_names(directory, experiment):
    """Get the filenames belonging to an experiment (LCM3 or LCM4)

    Args:
        directory ([string]): Path to the directory containing the mzid files
        experiment ([string]): "LCM3" or "LCM4"

    Returns:
        [type]: List of mzid file names (without the .mzid file extention)
    """
    file_names = []

    #Add the name of the files from "experiment" into the list
    for filename in os.listdir(directory):
        if regex.search(experiment,filename) and filename.endswith(".mzid"):
            file_names.append(filename[:-5])

    file_names.sort()
    return file_names

def extract_prot_xml(prot_xml_file):
    """This function returns the 1% FDR threshold of a prot.xml file and the dictionary with prbability of each protein as calculated by PP

    Args:
        prot_xml_file ([string]): Complete path + name to the prot.xml file

    Returns:
        [(float, dict)]: The float number is the 1% FDR threshold as defined by PP. The dic is the probability of each protein as calcualted by PP
    """

    #Exrract the prob threshold with 1% FDR
    file = codecs.open(prot_xml_file, "r", "utf-8")
    a = str(file.readlines())
    begining_of_prob = (a.find('<error_point error="0.01'))+38
    end_of_prob = (a.find('<error_point error="0.01'))+44
    threshold = float(a[begining_of_prob:end_of_prob])

    #Extract each protein's probability
    prot_threshold_dict = dict()
    a = pyteomics.protxml.DataFrame(prot_xml_file)
    for i in range(len(a)):
        prot_threshold_dict[a.iloc[i, 2]] = a.iloc[i, 0]
    
    return (threshold, prot_threshold_dict)


def raw_row_to_tuple(row_dict):
    """Parse a row of the mzid file (in dict format produced by pyteomics.mzid.read function), and returns a tupel with first element being protein accession, and second element a list with all the features

    Args:
        row_dict ([dict]): A dictionary contianing all the info from one row of mzid file

    Returns:
        [(string, list)]: String is the protein accession. The list contains all the features for that protein.
    """
    
    #retention_time = float(row_dict["scan start time"]) #Retention time in seconds

    #Rank 5 xcorr, otherwise the last rank
    Comet_xcorr_rank5 = row_dict["SpectrumIdentificationItem"][len(row_dict["SpectrumIdentificationItem"])-1]["Comet:xcorr"]
    # if len(row_dict["SpectrumIdentificationItem"])-1 != 4:
    #     print(row_dict["SpectrumIdentificationItem"][0]["PeptideSequence"], len(row_dict["SpectrumIdentificationItem"])-1)

            


    row_dict=row_dict["SpectrumIdentificationItem"][0]


    #Assing the 13 features we want in our dataset for the ranked 1 matches
    PeptideSequence = row_dict["PeptideSequence"] #need
    chargeState = row_dict["chargeState"] #need
    experimental_MZ = (row_dict["experimentalMassToCharge"])*chargeState - (chargeState*1.00727647) #need
    calculated_MZ = (row_dict["calculatedMassToCharge"])*chargeState - (chargeState*1.00727647)
    massdiff = experimental_MZ - calculated_MZ #need
    num_matched_peaks = row_dict["number of matched peaks"]
    num_unmatched_peaks = row_dict["number of unmatched peaks"]
    enz_C_terminus = PeptideSequence[-1] == "K" or  PeptideSequence[-1] == "R" #need
    enz_N_terminus = PeptideSequence[0] == "K" or  PeptideSequence[0] == "R" #need

    #Fraction of matched peaks
    if num_unmatched_peaks != 0:
        ion_fraction_matched = num_matched_peaks / num_unmatched_peaks #need
    else:
        ion_fraction_matched = 0

    Comet_xcorr = row_dict["Comet:xcorr"] #need
    Comet_deltacn = row_dict["Comet:deltacn"] #need
    #Comet_deltacnstar = row_dict["Comet:deltacnstar"]
    Comet_spscore = row_dict["Comet:spscore"] #need
    Comet_sprank = row_dict["Comet:sprank"] #need
    #Comet_exp_value = row_dict["Comet:expectation value"]
    accession = row_dict['PeptideEvidenceRef'][0]["accession"]
    

    #Comet deltaCN 5
    if Comet_xcorr !=0:
        Comet_deltacn_5 = (Comet_xcorr - Comet_xcorr_rank5)/Comet_xcorr #need
    else:
        Comet_deltacn_5 = 0



    #Add name of all the data into a dictionary except the "accession" value
    all_data ={
    #"retention_time":retention_time,
    "PeptideSequence":PeptideSequence,
    "chargeState":chargeState,
    "exp_mz":experimental_MZ,
    #"calculated_MZ":calculated_MZ,
    "xCorr":Comet_xcorr,
    "deltaCN":Comet_deltacn,
    "deltaCN_5": Comet_deltacn_5,
    #"Comet_deltacnstar":Comet_deltacnstar,
    "SpScore":Comet_spscore,
    "ln_SpRank":np.log(Comet_sprank),
    "massdiff": massdiff,
    "ionFrac": ion_fraction_matched,
    "enz_C_ter":enz_C_terminus,
    "enz_N_ter":enz_N_terminus
    #"Comet_exp_value":Comet_exp_value,
    #"num_matched_peaks":num_matched_peaks,
    #"num_unmatched_peaks":num_unmatched_peaks
    }

    #returns a tuple with the accesion name and the list of info
    return (accession,all_data)

class pixel():
    """A pixel object containing protein onjects in a list
    """

    __slots__ = ("name", "x", "y", "proteins")

    def __init__(self, name, protein_list):
        self.name = name #The name (unique number) of the pixel
        self.x = 0 #Initial x position
        self.y = 0 #Initial y position
        self.proteins = protein_list #List of proteins inside the pixel
    
    def len(self):
        return len(self.proteins)

    #Return the accession numbers of the proteins
    def get_protein_accession(self):
        return [protein.accession for protein in self.proteins]

    #Function to get the protein with a given accession in the pixel
    def __getitem__(self, name):
        for protein in self.proteins:
            if protein.accession == name:
                return protein
        return None


class peptide():
    """A peptide object
    """

    __slots__ = ("PeptideSequence","chargeState","xCorr", 
                "exp_mz","deltaCN","deltaCN_5",
                "SpScore","ln_SpRank", "massdiff",
                 "ionFrac", "enz_C_ter", "enz_N_ter")

    def __init__(self, data_dic):
        self.PeptideSequence = data_dic["PeptideSequence"]
        self.chargeState = data_dic["chargeState"]
        self.exp_mz = data_dic["exp_mz"]
        #self.calculated_MZ = data_dic["calculated_MZ"]
        #self.num_matched_peaks = data_dic["num_matched_peaks"]
        #self.num_unmatched_peaks = data_dic["num_unmatched_peaks"]
        self.xCorr = data_dic["xCorr"]
        self.deltaCN = data_dic["deltaCN"]
        self.deltaCN_5 = data_dic["deltaCN_5"]
        #self.Comet_deltacnstar = data_dic["Comet_deltacnstar"]
        self.SpScore = data_dic["SpScore"]
        self.ln_SpRank = data_dic["ln_SpRank"]
        self.massdiff = data_dic["massdiff"]
        self.ionFrac = data_dic["ionFrac"]
        self.enz_C_ter = data_dic["enz_C_ter"]
        self.enz_N_ter = data_dic["enz_N_ter"]
        #self.Comet_sprank = data_dic["Comet_sprank"]
        #self.Comet_exp_value = data_dic["Comet_exp_value"]
        #self.retention_time = data_dic["retention_time"]
    
    #Returns the features of the peptide as a list
    def get_features(self):
        return [self.PeptideSequence,
                self.chargeState,
                self.xCorr, 
                self.exp_mz,
                self.deltaCN,
                self.deltaCN_5,
                self.SpScore,
                self.ln_SpRank,
                self.massdiff,
                self.ionFrac,
                self.enz_C_ter,
                self.enz_N_ter
            ]

    def get_pep_sequence(self):
        return self.PeptideSequence

class protein():
    """A protein object
    """

    __slots__ = ("accession", "peptides", "one_perc_FDR_threshold", "prot_proph_prob")

    #accession, peptide_object_list, prob_threshold, prot_threshold_dic[accession]
    def __init__(self, accession, peptides, one_perc_FDR_threshold, prot_proph_prob):
        self.accession = accession #Accession number of the protein
        self.peptides = peptides #The peptide lists of the protein
        self.one_perc_FDR_threshold = one_perc_FDR_threshold #1% FDR threshold as defined by PP
        self.prot_proph_prob = prot_proph_prob #ProteinProphet probability of the protein
    
    def __len__(self):
        return len(self.peptides)

    #Number of times we see peptides associated to a protein
    def get_spectral_count(self):
        return len(self.peptides)
    
    def get_all_massdiff(self):
        return [peptide.massdiff for peptide in self.peptides]

    def get_all_xcorr(self):
        return [peptide.xCorr for peptide in self.peptides]
    
    def get_all_deltaCN(self):
        return [peptide.deltaCN for peptide in self.peptides]
    
    #Number of times we see UNIQUE peptides associated to a protein
    def get_peptide_count(self):
        unique_peptides = set()
        
        for peptide in self.peptides:
            unique_peptides.add(peptide.get_pep_sequence())
        
        return len(unique_peptides)

    
    def is_decoy(self):
        return decoy_prefix in self.accession

    def get_attr_values(self, *argv):
        """Returns the attribute values (mean, min, max, sd) of a protein as a dictionary

        Returns:
            [dict]: Returns all the attributes values requested as a dict
        """

        #Result dictionary to save the outputs
        result_dic = dict()

        for arg in argv:
            peptide_attr_list = [getattr(peptide, arg) for peptide in self.peptides] #Get the list of attributes belnging to the peptide

            #Get the mean, sd of the values in the list
            if arg != "chargeState":
                if len(peptide_attr_list) > 1:
                    result_dic[str("mean_"+arg+"_POI")] = np.mean(peptide_attr_list)#mean
                    result_dic[str("sd_"+arg+"_POI")] = statistics.stdev(peptide_attr_list) #Sd
                    result_dic["One_peptide_in_protein_POI"] = 0

                else:
                    result_dic[str("mean_"+arg+"_POI")] = peptide_attr_list[0] #mean
                    result_dic[str("sd_"+arg+"_POI")] = 0 #sd
                    result_dic["One_peptide_in_protein_POI"] = 1

                #Set the min and max of the values
                result_dic[str("max_"+arg+"_POI")] = max(peptide_attr_list)
                result_dic[str("min_"+arg+"_POI")] = min(peptide_attr_list)
            else:
                result_dic[str("mode_chargeState"+"_POI")]= statistics.mode(peptide_attr_list)


        return {**result_dic, **self.get_features_from_highest_xCorr_peptide()}


    def get_mean_deltaCN(self):
        deltaCN_list=[peptide.deltaCN for peptide in self.peptides]

        if len(deltaCN_list) > 1:
            return (sum(deltaCN_list))/len(deltaCN_list)
        else:
            return deltaCN_list[0]

    def get_features_from_highest_xCorr_peptide(self):
        max_xcorr = max([peptide.xCorr for peptide in self.peptides])

        for peptide in self.peptides:
            if peptide.xCorr == max_xcorr:
                return {
                    #"top_xCorr_chargeState_POI":peptide.chargeState,
                    "top_xCorr_exp_mz_POI": peptide.exp_mz,
                    "top_xCorr_xCorr_POI": peptide.xCorr,
                    "top_xCorr_deltaCN_POI": peptide.deltaCN,
                    "top_xCorr_deltaCN_5_POI":peptide.deltaCN_5,
                    "top_xCorr_SpScore_POI":peptide.SpScore,
                    "top_xCorr_ln_SpRank_POI": peptide.ln_SpRank,
                    "top_xCorr_massdiff_POI":peptide.massdiff,
                    "top_xCorr_ionFrac_POI": peptide.ionFrac,
                    "top_xCorr_enz_C_ter_POI":peptide.enz_C_ter,
                    "top_xCorr_enz_N_ter_POI": peptide.enz_N_ter
                }


    def get_SD_deltaCN(self):
        deltaCN_list=[peptide.deltaCN for peptide in self.peptides]
        
        if len(deltaCN_list) > 1:
            return statistics.stdev(deltaCN_list)
        # else:
        #     return 0


    def get_mean_xCorr(self):
        xCorr_list = [peptide.xCorr for peptide in self.peptides]

        if len(xCorr_list) > 1:
            return (sum(xCorr_list))/len(xCorr_list)
        else:
            return xCorr_list[0]

    def get_SD_xCorr(self):
        xCorr_list = [peptide.xCorr for peptide in self.peptides]

        if len(xCorr_list) > 1:
            return statistics.stdev(xCorr_list)
        # else:
        #     return 0


    def get_highest_xCorr(self):
        return max([peptide.xCorr for peptide in self.peptides])


    def get_highest_deltaCN(self):
        return max([peptide.deltaCN for peptide in self.peptides])
    

def file_to_pixel(raw_data_file_name,directory):
    """Parse mzid files and return list of protein objects in that mzid file (which represents a pixel)

    Args:
        raw_data_file_name ([string]): name of the mzid file (without the .mzid extension)
        directory ([string]): Path to the mzid file

    Returns:
        [pixel]: list of protein objects inside the mzid file as a pixel object
    """

    #Read the mzid file
    raw_data = pyteomics.mzid.read(directory+raw_data_file_name+".mzid")

    #Dictionary with the accession as the keys, 
    #and the list of lists with peptide informations 
    protein_data_dic = dict() 

    #Parse each row of the mzid file
    for row in raw_data:
        protein_info = raw_row_to_tuple(row)


        #if the key "accesion" is already in the data dictionary
        #append the new peptide info to it
        #Else, make a new list to add the info
        if protein_info[0] in protein_data_dic:
            protein_data_dic[protein_info[0]].append(protein_info[1])
        else:
            protein_data_dic[protein_info[0]]=[protein_info[1]]
    

    #Add the protein prophet info to the proteins in the pixel
    prot_proph_info = extract_prot_xml(directory+raw_data_file_name+".prot.xml")

    protein_list = dic_to_proteins(protein_data_dic, prot_proph_info) #List of the protein objects in a pixel 

    return pixel(raw_data_file_name[31:33], protein_list) #Return the pixel object


def dic_to_proteins(accession_data_dic, prot_proph_info):
    """Helper function receives the dictionary produced by the file_to_pixel function, and returns the protein objects in a list

    Args:
        accession_data_dic ([dict]): The dictionary produced by the file_to_pixel function
        prot_proph_info ([(int, dict))]): The 1% FDR threshold, and the dictionaru containing info about the proteins

    Returns:
        [list]: list of protein objects
    """
    protein_list = []

    prob_threshold = prot_proph_info[0]  # 1% FDR threshold probability
    prot_threshold_dic =  prot_proph_info[1]  #Dictionary with keys being protein accession and value being their threshold

    for accession in accession_data_dic:
        peptides_list = accession_data_dic[accession] #List of peptides for a given accession
        peptide_object_list = [] #List to store peptide objects

        for peptide_data in peptides_list:
            peptide_object_list.append(peptide(peptide_data))
        
        if accession in prot_threshold_dic:
            prot_threshold = prot_threshold_dic[accession]
        else:
            prot_threshold = 0 #Set the probability of the proteins not in the ProteinProphet to zero 
        protein_list.append(protein(accession, peptide_object_list, prob_threshold, prot_threshold))

    return protein_list


def createPixel(file, directory, coord_dic):
    """Helper function to make multiprocessing possible by reading each pixel seperately from the mzid file.

    Args:
        file ([string]): name of the mzid file (without the .mzid extension)
        directory ([string]): Path to the directory containing the mzid file
        coord_dic ([dict]): Dictionary containing the x and y location of each pixel

    Returns:
        [pixel]: pixel object
    """
    start_time = time.time()
    a_pixel = file_to_pixel(file, directory)
    a_pixel.x = coord_dic[int(a_pixel.name)][0]
    a_pixel.y = coord_dic[int(a_pixel.name)][1]
    
    print("Pixel ", a_pixel.name, "took ", time.time()-start_time, "sec")
    return a_pixel


def set_pixels_into_dataset(file_names, directory, coord_dic):
    """This function receives file names associated with the pixels and returns list of pixels objects

    Args:
        file_names ([list]): List of file names (without extensions) associated to the pixels
        directory ([string]): Path to the directory containing the files
        coord_dic ([type]): Dictionary of the pixels and thier coordinates

    Returns:
        [type]: [description]
    """
        
    with Pool(cpu_count()) as pool:
        res = pool.map(partial(createPixel, directory=directory, coord_dic=coord_dic), file_names)
        
    return list(res)


class MSI_dataset:
    """MSI dataset object definition
    """

    __slots__ = ("name", "nrow", "ncol", "coords", "pixels")

    def __init__(self, name, nrow, ncol, directory, first_pixel_num, first_pixel_pos):
        self.name = name #Name of the dataset
        self.nrow = nrow #Number of the row of the dataset
        self.ncol = ncol #Number of the columns in the dataset
        self.coords = self.pixel_to_coord(first_pixel_num, nrow,ncol, first_pixel_pos)
        self.pixels = set_pixels_into_dataset(get_file_names(directory,name), directory, self.coords) #List of the pixel objects in the dataset
    
    def __len__(self):
        return int(self.nrow * self.ncol)

    #Returns the dictionary with the pixel coordinates in order
    def pixel_to_coord(self, lowest_num, nrow, ncol, first_num_pos):
        coordinates = dict()

        if (first_num_pos == "top_right"):
            for i in range(nrow):
                for j in range((ncol-1),-1,-1):
                    coordinates[lowest_num]=(j,i)
                    lowest_num+=1
            return coordinates
        elif (first_num_pos == "bottom_left"):
            for i in range((nrow-1),-1,-1):
                for j in range(ncol):
                    coordinates[lowest_num]=(j,i)
                    lowest_num+=1
            return coordinates

    #This is to define a way to access each pixel by their coordinates
    def __getitem__(self, pos):
        x,y = pos
        for pixel in self.pixels:
            if pixel.x ==x and pixel.y==y:
                return pixel


def get_adj_pixel_coord(pixel, dataset):
    """Function that receives a pixel object, and returns a list of tuples with coordinate of the adjacent pixels with respect to the dataset

    Args:
        pixel ([pixel]): A pixel object
        dataset ([MSI_dataset]): A MSI-dataset object

    Returns:
        [list]: List of tuples with coordinate of the adjacent pixels
    """
    x = pixel.x
    y = pixel.y
    pixel_coord_list = [
        (x-1,y-1),(x,y-1),(x+1,y-1),
        (x-1,y),          (x+1, y),
        (x-1,y+1),(x,y+1),(x+1,y+1)
    ]

    valid_pixel_list = [] #Valid pixels list

    #Remove invalid coordinates
    possible_coords = (dataset.coords.values())
    for a_pixel in pixel_coord_list:
        if a_pixel in possible_coords:
            valid_pixel_list.append(a_pixel)

    return valid_pixel_list


def adj_features(protein_name, POI, dataset):
    """Function receiving a pixel, protein_name, and a dataset, then returns the features of interest from the adjacnet pixels for a protein

    Args:
        protein_name ([string]): Accession number of a protein
        POI ([pixel]): A pixel-of-interest
        dataset ([MSI_dataset]): A MSI dataset object that contains the POI

    Returns:
        [dict]: A dicitonary containing the features of interest from the adjacent pixels for a protein
    """

    adj_pixel_coord = get_adj_pixel_coord(POI, dataset) #Get the list of adjacent pixel coordinates of the pixel of interest (POI)

    #Get the three highest Xcorr , deltaCN, protein prophet probabilities, and spectral count
    #If protein info are lacking in the adjacent pixels, substitute the feature with zero
    highest_xCorr = []
    highest_deltaCN = []
    #highest_protein_prob = []
    highest_spectral_count = []

    all_xCorr = []
    all_deltaCN = []
    all_massdiff = []
    all_specCount = []
    

    for coord in adj_pixel_coord:
        an_adj_pixel = dataset[coord[0],coord[1]] #Get one adjacent pixel object

        
        #print(protein_name, POI.name, an_adj_pixel.name)
        
        if protein_name in an_adj_pixel.get_protein_accession():
            protein_obj = an_adj_pixel[protein_name]
        
            highest_xCorr.append(protein_obj.get_highest_xCorr())
            highest_deltaCN.append(protein_obj.get_highest_deltaCN())
            #highest_protein_prob.append(protein_obj.prot_proph_prob)
            highest_spectral_count.append(protein_obj.get_spectral_count())

            all_xCorr.append(protein_obj.get_all_xcorr())
            all_deltaCN.append(protein_obj.get_all_deltaCN())
            all_massdiff.append(protein_obj.get_all_massdiff())
            all_specCount.append(protein_obj.get_spectral_count())
            
        else: 
            highest_xCorr.append(0)
            highest_deltaCN.append(0)
            #highest_protein_prob.append(0)
            highest_spectral_count.append(0)
        

    highest_xCorr.sort(reverse=True)
    highest_deltaCN.sort(reverse=True)
    #highest_protein_prob.sort(reverse=True)
    highest_spectral_count.sort(reverse=True)

    spatial_info_not_exist = 1 if len((all_specCount))==0 else 0

    num_surrounding_pixels_with_PSM = len(all_xCorr)
    
    avg_surrounding_SPCount = np.mean(highest_spectral_count)


    if len(list(itertools.chain(*all_xCorr))) == 0:
        all_xCorr = 0
    else:
        all_xCorr = np.mean(list(itertools.chain(*all_xCorr)))
    

    if len(list(itertools.chain(*all_deltaCN))) == 0:
        all_deltaCN = 0
    else:
        all_deltaCN = np.mean(list(itertools.chain(*all_deltaCN)))
    

    if len(list(itertools.chain(*all_massdiff))) == 0:
        all_massdiff = 0
    else:
        all_massdiff = np.mean(list(itertools.chain(*all_massdiff)))


    if len((all_specCount)) == 0:
        all_specCount = 0
    else:
        all_specCount = np.mean(all_specCount)
    



    return {
        "highest_xCorr":highest_xCorr[0:3],
        "highest_deltaCN":highest_deltaCN[0:3],
        #"highest_protein_prob":highest_protein_prob[0:3],
        "spatial_info_not_exist": spatial_info_not_exist,
        "highest_spectral_count": highest_spectral_count[0:3],
        "mean_surrounding_spectral_count": avg_surrounding_SPCount,
        "mean_all_surrounding_spectral_count": all_specCount,
        "mean_all_surrounding_xCorr": all_xCorr,
        "mean_all_surrounding_deltaCN": all_deltaCN,
        "mean_all_surrounding_massdiff": all_massdiff,
        "num_surrounding_pixels_with_PSM": num_surrounding_pixels_with_PSM
    }


    

def get_pixel_data(pixel, MSI_dataset):
    """Function extracting the necessary information from a pixel object on a MSI_dataset

    Args:
        pixel ([pixel]): A pixel object
        MSI_dataset ([MSI_dataset]): A MSI_dataset containing the pixel

    Returns:
        [list]: list of features from a pixel
    """

    List_of_results = []

    protein_names = pixel.get_protein_accession() #Get the proteins that exist on the pixel of interest (POI)

    for protein_accession in protein_names:

        prot_obj_POI = pixel[protein_accession] #Protein object on the POI

        #Get the features on the POI
        spectral_count_POI = prot_obj_POI.get_spectral_count() #Get Spectral count on POI
        peptide_count_POI = prot_obj_POI.get_peptide_count() #Get peptide count on POI
    

        #Get the features on the POI using the new method
        combined_peptide_result_dic = prot_obj_POI.get_attr_values("xCorr","deltaCN","deltaCN_5",
                "SpScore","ln_SpRank", "massdiff","chargeState",
                 "ionFrac")
        
        
        #NEW ADDITION TO VER2
        protein_proph_prob = prot_obj_POI.prot_proph_prob

        
        #Get the features on the adjacent pixels
        adjacent_features = adj_features(protein_accession, pixel, MSI_dataset)

        
        #Get the target
        target = 0
        if prot_obj_POI.is_decoy():
            target = 0
        elif (prot_obj_POI.prot_proph_prob < prot_obj_POI.one_perc_FDR_threshold):
            target = 0
        elif (prot_obj_POI.prot_proph_prob >= prot_obj_POI.one_perc_FDR_threshold):
            target = 1


        
        #print(pixel.name)
        List_of_results.append( {**combined_peptide_result_dic,**{
            "accession":protein_accession,
            "protein_proph_prob": protein_proph_prob,
            "POI_name": pixel.name,
            #"mean_xCorr_POI":mean_xCorr_POI,
            #"sd_xCorr_POI":sd_xCorr_POI,
            #"mean_deltaCN_POI":mean_deltaCN_POI,
            #"sd_deltaCN_POI":sd_deltaCN_POI,
            "spectral_count_POI":spectral_count_POI,
            "peptide_count_POI":peptide_count_POI,
            #"max_xCorr_POI": max_xCorr_POI,
            #"max_deltaCN_POI": max_deltaCN_POI,
            "top_xCorr_1":adjacent_features["highest_xCorr"][0],
            "top_xCorr_2":adjacent_features["highest_xCorr"][1],
            "top_xCorr_3":adjacent_features["highest_xCorr"][2],
            "top_deltaCN_1":adjacent_features["highest_deltaCN"][0],
            "top_deltaCN_2":adjacent_features["highest_deltaCN"][1],
            "top_deltaCN_3":adjacent_features["highest_deltaCN"][2],
            # "top_pro_proph_1":adjacent_features["highest_protein_prob"][0],
            # "top_pro_proph_2":adjacent_features["highest_protein_prob"][1],
            # "top_pro_proph_3":adjacent_features["highest_protein_prob"][2],
            "top_SpecCount_1":adjacent_features["highest_spectral_count"][0],
            "top_SpecCount_2":adjacent_features["highest_spectral_count"][1],
            "top_SpecCount_3":adjacent_features["highest_spectral_count"][2],
            "mean_surrounding_spectral_count":adjacent_features["mean_surrounding_spectral_count"],
            "mean_all_surrounding_xCorr":adjacent_features["mean_all_surrounding_xCorr"],
            "mean_all_surrounding_deltaCN":adjacent_features["mean_all_surrounding_deltaCN"],
            "mean_all_surrounding_massdiff":adjacent_features["mean_all_surrounding_massdiff"],
            "mean_all_surrounding_spectral_count": adjacent_features["mean_all_surrounding_spectral_count"],
            "num_surrounding_pixels_with_PSM": adjacent_features["num_surrounding_pixels_with_PSM"],
            "spatial_info_not_exist":adjacent_features["spatial_info_not_exist"],
            "target":target

        }})

    print("Pixel ",pixel.name, " is done ", "Feature engineering")
    return List_of_results   



def output_dataset(a_MSI_dataset, output_dir, name):
    """Function that uses multiprocessing to process a MSI_dataset and output it in the output directory

    Args:
        a_MSI_dataset ([MSI_dataset]): A MSI_dataset
        output_dir ([string]): The output directory to store the tabular csv files for machine learning
        name ([string]): Name of the csv file to be saved
    """

    with Pool(cpu_count()) as pool:
        final_list = pool.map(partial(get_pixel_data, MSI_dataset=a_MSI_dataset), a_MSI_dataset.pixels)
    #final_list = map(partial(get_pixel_data, MSI_dataset=a_MSI_dataset), a_MSI_dataset.pixels)

    final_list = list(itertools.chain(*final_list))  
    data = pd.DataFrame(final_list)
    data.to_csv (output_dir+name+"_BIG_"+".csv", index = False, header=True)

def main():
    #Get the ratios for downsampling
    ratios = ["0.00","0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90"]

    for ratio in ratios:
        directory = PP_xml_mzid_dir+"DSratio_"+ratio+"/"
        print("Reading the files in", directory)
        print("\n")
        print("Preparing the TRAINING data ...")
        time1 = time.time()
        training_data = MSI_dataset(name="LCM3", nrow=6, ncol=4, directory = directory, first_pixel_num = 37, first_pixel_pos = "top_right")
        print("Creating the TRAINING dataset took ", time.time()-time1)
        print("\n")

        # with open(directory+'MSIdataset_2021_03_07_LCM3_BIG.pkl', 'wb') as output:
        #     pickle.dump(training_data, output, pickle.HIGHEST_PROTOCOL)

        #training_data = pickle.load( open(directory+"MSIdataset_2021_03_01_LCM3_BIG.pkl", "rb" ) )
        
        print("pickled completed")

        time1 = time.time()
        print("Feature engineering and outputting them ....")
        output_dataset(training_data, output_spatial_data_dir, "training_"+ratio)
        print("Feature engineering took ", time.time()-time1)
        print("\n")
        del(training_data)


        print("Preparing the TESTING data ...")
        time1 = time.time()
        testing_data = MSI_dataset(name="LCM4", nrow=6, ncol=4, directory = directory, first_pixel_num = 61, first_pixel_pos = "bottom_left")
        print("Creating the TESTING dataset took ", time.time()-time1)
        print("\n")

        # with open(directory+'MSIdataset_2021_01_21_LCM4.pkl', 'wb') as output:
        #     pickle.dump(testing_data, output, pickle.HIGHEST_PROTOCOL)

        time1 = time.time()
        print("Feature engineering and outputting them ....")
        output_dataset(testing_data, output_spatial_data_dir, "testing_"+ratio)
        print("Feature engineering took ", time.time()-time1)
        print("\n")
        del(testing_data)

        print("------------------------------------------------------------")

if __name__ == "__main__":
    main()