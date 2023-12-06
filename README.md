# [Improving Protein Identification In Mass Spectrometry Imaging Using Machine Learning and Spatial Spectral Information](http://dx.doi.org/10.20381/ruor-27364)

## Description
Mass spectrometry imaging (MSI) is a high-throughput technique that in addition to performing protein identification, can capture the spatial localization of proteins within biological tissue. Nevertheless, sample pre-processing and MSI instrumentation limit protein identification capability in MSI compared to more standard tandem mass spectrometry-based proteomics methods. Despite these limitations, the current protein identification approaches used in MSI were originally designed for standard mass spectrometry-based proteomics and do not take advantage of the spatial information acquired in MSI. Herein, I explore the benefit of using the spatial spectral information for protein identification using two objectives. For the first objective, I developed a novel supervised learning spatially-aware protein identification algorithm for mass spectrometry imaging (SAPID-MSI) and benchmarked it against ProteinProphet and Percolator, which are state-of-the-art tools for protein identification confidence assessment. I showed that SAPID-MSI identifies on average 20% more proteins at a 1% false discovery rate compared to the other two algorithms. Furthermore, more proteins are identified when spatial features are used to identify proteins compared to when they are not suggesting their additional benefit. For the second objective, I used SAPID-MSI to rescue false positive and false negative protein identifications made by ProteinProphet. By examining a combination of data sampling and learning algorithms, I was able to achieve a good classification performance compared to the baseline given the extreme imbalance in the dataset. Finally, by improving proteome characterization in MSI, our approach will help providing a better understanding of the processes taking place in biological tissues.

## Prerequisites
### Python Environment
[Conda](https://docs.conda.io/) was used to manage python packages and their dependencies. To replicate the environment (sapid-env) used to produce the results of this research, run the following command:
```
conda env create -f environment.yml
```
Then activate the environment using the following command:
```
conda activate sapid-env
```

### External Software and Datasets
The following softwares and datasets needs to be installed and saved prior to running the codes:

* [The Trans-Proteomic Pipeline (TPP) version 5.2.0 to run ProteinProphet, msconvert tool, and Comet (version 2018.01 rev.4)](http://tools.proteomecenter.org/wiki/index.php?title=Software:TPP)
* [Crux (version 3.2) to run Percolator (version 3.05.nightly-106-2c8457a7)](https://crux.ms/download.html)
* [Protein sequence database](https://massive.ucsd.edu/ProteoSAFe/dataset_files.jsp?task=06750fe8bf7e437b87893a21a931f99a#%7B%22table_sort_history%22%3A%22main.collection_asc%22%2C%22main.collection_input%22%3A%22sequence%7C%7CEXACT%22%7D)
* [RAW mass spectrometry data used for training (LCM3 experiment)](https://massive.ucsd.edu/ProteoSAFe/dataset_files.jsp?task=06750fe8bf7e437b87893a21a931f99a#%7B%22table_sort_history%22%3A%22main.collection_asc%22%2C%22main.file_descriptor_input%22%3A%22LCM3%22%2C%22main.collection_input%22%3A%22raw%7C%7CEXACT%22%7D)
* [RAW mass spectrometry data used for testing (LCM4 experiment)](https://massive.ucsd.edu/ProteoSAFe/dataset_files.jsp?task=06750fe8bf7e437b87893a21a931f99a#%7B%22table_sort_history%22%3A%22main.collection_asc%22%2C%22main.file_descriptor_input%22%3A%22LCM4%22%2C%22main.collection_input%22%3A%22raw%7C%7CEXACT%22%7D)

## Usage
Following python files (.py) need to be configured and run in the following order:
1. **Constants:** The path and constants used in SAPID project. Needs to be imported in all the modules.
2. **DownSample:** This module downsamples the pep.xml files from Comet.
3. **mzID_TPP:** This module converts each comet pep.xml into mzID and store them in their respective folder. Then, it runs ProteinProphet on each of the mzID files.
4. **CreateDataset:** This module performs feature engineering and creates tabular datasets with spatial and tabular features. <strong>❗The code parallelize this process on all of the cores in your machine. Change the number of cores in function `set_pixels_into_dataset()` if needed.</strong>
5. **percolator_run:** This mudule uses Crux command line to run Percolator on the downsampled pep.xml files.
6. **feature_names:** The feature sets and feature names used in SAPID and MUST comply with the ones used in __3_CreateDataset.py__ module. 
7. **SAPID:** Performing machine learning experiments as outlined in Chapter 3 of my Master's thesis.
8. **FalseRescue:** Performing machine learning experiments with data sampling as outlined in Chapter 4 of my Master's thesis.




 


