#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Soroush Shahryari Fard
# =============================================================================
"""The constants used in SAPID project. Needs to be imported in all the modules."""
# =============================================================================

#Original Comet files are stored here. We only use this file for downsampling purposes, and getting the original
#file names (ie: no TPP or mzid conversion is performed on them)
comet_pep_xml_dir = "C:/TPP/data/soroush/Comet_PEP_XML/"

#The downsampled Comet files are stored here. We run mzid conversion and TPP on them
comet_downsampled_output_dir = "C:/TPP/data/soroush/downsampled_comet/"

#The mzid converted pep.xml and result of PeptideProphet and ProteinProphet is stored here
PP_xml_mzid_dir = "C:/TPP/data/soroush/PPxml_mzid_files/"

#The number of donwsampling ratios we want in [0.00, 0.99]
num_of_DSratios = 5

#The decoy prefix used for database search
decoy_prefix = "DECOY_"

#Output spatial data directory
output_spatial_data_dir = "C:/TPP/data/soroush/output_spatial_data/"

#Percolator output dir
percolator_output_dir = "C:/TPP/data/percolator/"

#The protein database address for percolator
database_address = "C:/TPP/data/dbase/M_musculus_2016-12-20_DECOY.fasta"