Computing PSSM files
=============================

iScore does not provide the PSSM files that are needed for the definition of the graph. However we have developped another tool that allows to get the PSSM in the desired format. This tool can be found at

https://github.com/DeepRank/PSSMGen

More information can be foud on the link above. PSSMGen is a simple python program using `Biopython` and `pdb2sql` to create and format the PSSM files. The only input needed is the PDB file of the decoy. To compute the PSSM file one can simply use :


>>> from pssmgen.pssm import PSSMdecoy
>>>
>>> gen = PSSMdecoy('1AK4')
>>>
>>> # generates the FASTA query
>>> gen.get_fasta()
>>>
>>> # generates the PSSM
>>> gen.get_pssm()
>>>
>>> # map the pssm to the pdb
>>> gen.map_pssm()