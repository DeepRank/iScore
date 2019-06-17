Computing PSSM files
=============================

As a preprocessing step the user must compute the PSSM files corespondng to the PDB files in the training/testing dataset. This can be acheived with the PSI-Blast library (https://ncbiinsights.ncbi.nlm.nih.gov/2017/10/27/blast-2-7-1-now-available/). The library BioPython allows an easy use of these libraries.


iScore contains a wrapper that allows to compute the PSSM data, map them to the PDB files and format them for further processing. The only input needed is the PDB file of the decoy. To compute the PSSM file one can simply use :


>>> from iScore.pssm.pssm import PSSM
>>>
>>> gen = PSSM(caseID = '1AK4', pdb_dir ='1AK4/pdb')
>>>
>>> # generates the FASTA query
>>> gen.get_fasta()
>>>
>>> # configure the generator
>>> gen.configure(blast=<path to blast binary>, database=<path to the blast db>)
>>>
>>> # generates the PSSM
>>> gen.get_pssm()
>>>
>>> # map the pssm to the pdb
>>> gen.map_pssm()
