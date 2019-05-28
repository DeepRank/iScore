#!/usr/bin/env python

"""
Map the the sequence of PDB file to that of PSSM file to get consistent sequence(no gap, no residue X),
and output mapped PSSM file and/or PDB file with the consisitent sequence.

Usage: python map_pssm2pdb.py <input PSSM file> <input PDB file> <ChainID of PDB file> <Output path>
Example: python map_pssm2pdb.py  ./pssm/4CPA.A.pssm  ./pdb/4CPA.pdb  A  ./test

Author: {0} ({1})
"""

import os
import sys
import numpy as np
from Bio import pairwise2

__author__ = "Cunliang Geng"
__email__ = "gengcunliang AT gmail.com"
USAGE = __doc__.format(__author__, __email__)


def check_input(args):
    """Validate user input

    Arguments:
        args {tuple} --  user input arguments
    """
    if len(args) != 4:
        sys.exit(USAGE)


def get_seq_resi_from_pdb(fpdb, chainID):
    """Get FASTA sequence and residue ID from PDB file.

    Arguments:
        fpdb {str} -- input pdb file
        chainID {str} -- the chain to get the sequence and residue number

    Raises:
        ValueError -- ChainID not exist in the pdb file

    Returns:
        list - sequence of the specific chain
        list - residue ID of the specific chain
    """

    res_codes = [
        # 20 canonical amino acids
        ('CYS', 'C'), ('ASP', 'D'), ('SER', 'S'), ('GLN', 'Q'),
        ('LYS', 'K'), ('ILE', 'I'), ('PRO', 'P'), ('THR', 'T'),
        ('PHE', 'F'), ('ASN', 'N'), ('GLY', 'G'), ('HIS', 'H'),
        ('LEU', 'L'), ('ARG', 'R'), ('TRP', 'W'), ('ALA', 'A'),
        ('VAL', 'V'), ('GLU', 'E'), ('TYR', 'Y'), ('MET', 'M'),
        # Non-canonical amino acids
        ('ASX', 'B'), ('SEC', 'U'), ('GLX', 'Z'),
        # ('MSE', 'M'), ('SOC', 'C'),
        # Canonical xNA
        ('  U', 'U'), ('  A', 'A'), ('  G', 'G'), ('  C', 'C'),
        ('  T', 'T'),
    ]

    three_to_one = dict(res_codes)
    _records = set(['ATOM  ', 'HETATM'])

    chainID = chainID.upper()
    sequence = []
    resID = []
    chains = set()
    read = set()
    with open(fpdb, "r") as f:
        for line in f:
            line = line.strip()
            if line[0:6] in _records:
                resn = line[17:20]
                chain = line[21]
                resi = line[22:26]
                icode = line[26]
                r_uid = (resn, chain, resi, icode)
                chains.add(chain)
                if chain == chainID:
                    if r_uid not in read:
                        read.add(r_uid)
                    else:
                        continue
                    aa_resn = three_to_one.get(resn, 'X')
                    sequence.append(aa_resn)
                    resID.append(resi)
        if chainID not in chains:
            raise ValueError(
                "Chain `{}` NOT exist in PDB file '{}'".format(chainID, fpdb))

    return sequence, resID


def get_pssm(fpssm):
    """Get the content of PSSM file.

    Arguments:
        fpssm {str} -- input pssm file

    Raises:
        ValueError -- the line with number of columns not equal to 44

    Returns:
        [2D list] -- pssm
    """
    rule = tuple([str(i) for i in range(10)])
    pssm = []
    with open(fpssm, "r") as f:
        for line in f.readlines():
            line_raw = line
            line = line.strip()
            # only select lines that contain pssm values
            if line.startswith(rule):
                    # TODO parse pssm based on column index
                    # normal PSSM line have 44 columns. Abnormal <44 due to lakcing of gap between numbers.
                if len(line.split()) == 44:
                    pssm.append(line.split())
                else:
                    raise ValueError(
                        "Wrong format of the following line in PSSM file {}:\n{}".format(fpssm, line_raw))

    return pssm


def get_aligned_sequences(seq1, seq2):
    """Align two sequnces using global alignment and return aligned sequences.
        Paramters of global alignment:
            match: 1
            mismtach: 0
            gap open: -2
            gap extend: -1

    Arguments:
        seq1 {str} -- 1st sequence.
        seq2 {str} -- 2nd sequence.

    Returns:
        [numpy array] -- seq1_ali, aligned sequence for seq1
        [numpy array] -- seq2_ali, aligned sequence for seq1
    """

    ali = pairwise2.align.globalxs(seq1, seq2, -2, -1)
    seq1_ali = np.array([i for i in ali[0][0]])
    seq2_ali = np.array([i for i in ali[0][1]])

    return seq1_ali, seq2_ali


def write_pdb_remove_residue(fipdb, fopdb, chainID, resID):
    """Write PDB file with removing some residues.

    Arguments:
        fipdb {str} -- Input PDB file.
        fopdb {str} -- Output PDB file.
        chainID {str} -- The ID of the chain that the to-be-removed residues locates.
        resID {list} -- A list of residue ID to remove.
    """

    fout = open(fopdb, "w")
    resID = [str(i) for i in resID]
    _records = set(['ATOM  ', 'HETATM'])
    with open(fipdb, "r") as f:
        for line in f:
            if line[0:6] in _records:
                chain = line[21]
                resi = line[22:26].strip()
                if chain == chainID and resi in resID:
                    continue
                else:
                    fout.write(line)
            else:
                fout.write(line)
    fout.close()


def write_mapped_pssm_pdb(fpssm, fpdb, chainID, outdir):
    """Map PDB sequence to PSSM sequence to get the consistent sequence,
    and output mapped PSSM and/or PDB file with consistent sequence.

    Arguments:
        fpssm {str} -- input PSSM file
        fpdb {str} -- input PDB file
        chainID {str} -- the specific chain of PDB used to be mapped with PSSM, e.g. A or a
        outdir {str} -- path for output pssm file, e.g. /home/tes

    Output:
        mapped PSSM file:  pdbfilename.chainID.pdb.pssm
        mapped PDB file:   pdbfilename.chainID.pssm.pdb, only output when sequence of input PDB longer than consistent sequence.
    """
    # get pssm and pdb file name
    pdbname = os.path.basename(fpdb)
    pssmname = os.path.basename(fpssm)
    # get pdb sequence and residue numbers
    pdb_seq, pdb_resn = get_seq_resi_from_pdb(fpdb, chainID)
    pdb_seq_str = "".join(pdb_seq)
    # get pssm content and sequnce
    pssm = np.array(get_pssm(fpssm))
    pssm_seq_str = "".join(pssm[:, 1])

    # get aligned seqeuences
    pdb_seq_align, pssm_seq_align = get_aligned_sequences(pdb_seq_str, pssm_seq_str)

    # get indexes for matched and mismatched residues.
    index_match = pdb_seq_align == pssm_seq_align
    index_mismatch = np.logical_not(index_match)

    # make a gap sequence (only "-") and X sequence (only "X") that have same length as pdb/pssm_seq_align
    seqlen = len(pdb_seq_align)
    gap_seq = np.array(["-"] * seqlen)
    resX_seq = np.array(["X"] * seqlen)

    # get index of gap and residue X
    index_gappdb = gap_seq == pdb_seq_align
    index_resXpdb = resX_seq == pdb_seq_align
    index_gappssm = gap_seq == pssm_seq_align
    index_resXpssm = resX_seq == pssm_seq_align
    # get index of normal residues (not gap, not res X) for each sequence
    index_norm_pdb = np.logical_not(np.logical_or(index_gappdb, index_resXpdb))
    index_norm_pssm = np.logical_not(np.logical_or(index_gappssm, index_resXpssm))
    # get index of normal residues for both sequences
    index_norm = np.logical_and(index_norm_pdb, index_norm_pssm)
    # get index of mutated normal residues
    index_mut = np.logical_and(index_mismatch, index_norm)

    # raise warning for mutated normal residues
    if len(set(index_mut)) > 1:
        mut_seq = []
        for i in index_mut:
            if i:
                mut_seq.append("^")
            else:
                mut_seq.append("_")
        try:
            raise Warning("Warning: Mutations exist in following sequences:\n>{pdbname}_{chainid}:\n{pdbseq}\n>{pssmname}:\n{pssmseq}\n{mutseq}\n".format(
                pdbname=pdbname, chainid=chainID, pssmname=pssmname, pdbseq="".join(pdb_seq_align), pssmseq="".join(pssm_seq_align), mutseq="".join(mut_seq)))
        except Warning as e:
            print(e)

    # get pssm with index of normal residues for both sequences, this is the mapped pssm.
    index_norm_nogappssm = index_norm[np.logical_not(index_gappssm)]
    pssm_norm = pssm[index_norm_nogappssm]

    # add the residue number and name of PDB file to the mapped pssm
    # for pssm content, only keep the scoring matrix and information content
    header = ["pdbresi", "pdbresn", "seqresi", "seqresn", "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "IC"]
    header = np.transpose(np.array([[i] for i in header]))
    pdb_resn = [[i] for i in pdb_resn]
    pdb_seq = [[i] for i in pdb_seq]
    index_norm_nogappdb = index_norm[np.logical_not(index_gappdb)]
    resi_pdb = np.array(pdb_resn)[index_norm_nogappdb]
    resn_pdb = np.array(pdb_seq)[index_norm_nogappdb]
    pssm_out = np.concatenate((resi_pdb, resn_pdb, pssm_norm[:, :22], pssm_norm[:, -2:-1]), axis=1)
    pssm_out = np.concatenate((header, pssm_out))

    # write mapped pssm to file which is named with input PDB file name, chain ID and ".pdb.pssm"
    fopssm = os.path.join(outdir, os.path.splitext(pdbname)[0] + "." + chainID.upper() + ".pdb.pssm")
    with open(fopssm, "w") as f:
        for i in pssm_out:
            tmp1 = ["{:>7s}".format(j) for j in i[:4]]
            tmp2 = ["{:>4s}".format(j) for j in i[4:]]
            f.write(" ".join(tmp1+tmp2) + "\n")

    # write mapped PDB file if some residues not exist in the mapped PSSM file
    index_toremove = np.logical_or(np.logical_or(index_gappssm, index_resXpssm), index_resXpdb)
    index_gappdb_resXpssm = np.logical_and(index_gappdb, index_resXpssm)
    index_toremove = np.logical_and(index_toremove, np.logical_not(index_gappdb_resXpssm))
    fopdb = os.path.join(outdir, os.path.splitext(pdbname)[0] + "." + chainID.upper() + ".pssm.pdb")
    if len(np.unique(index_toremove)) == 2:
        # write PDB with removing some resdiues
        index_toremove_nogappdb = index_toremove[np.logical_not(index_gappdb)]
        resn_pdb_remove = np.array(pdb_resn)[index_toremove_nogappdb]
        resn_pdb_remove = [ i.strip() for i in resn_pdb_remove[:,0].tolist() ]
        write_pdb_remove_residue(fpdb, fopdb, chainID, resn_pdb_remove)

        print("Warning: A new PDB file with all chains and consistent sequence of chain {} is generated: {}\n".format(chainID, fopdb))


if __name__ == "__main__":
    check_input(sys.argv[1:])
    fpssm, fpdb, chainID, outdir = sys.argv[1:]
    # fpssm, fpdb, chainID, outdir = "4CPA.A.pssm", "4CPA.pdb", "A",  "."
    write_mapped_pssm_pdb(fpssm, fpdb, chainID, outdir)
