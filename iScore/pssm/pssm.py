import os, glob, shutil
from Bio.Blast.Applications import NcbipsiblastCommandline
from pdb2sql.pdb2sqlcore import pdb2sql

from iScore.pssm.map_pssm2pdb import write_mapped_pssm_pdb


class PSSM(object):

    def __init__(self,caseID='',pdbdir='pdb'):

        """Compute the PSSM and map the the sequence for a series of decoys.

        Args:
            caseID (TYPE): Name of the case. This must correspond to a directory name
            pdbdir (str, optional): directory name where the decoys pdbs are stored within caseID

        Example:

        >>> # import the module
        >>> from iScore.pssm.pssm import PSSM
        >>>
        >>> # create ab instance
        >>> gen = PSSM('1AK4',pdbdir='water')
        >>>
        >>> # configure the generator
        >>> gen.configure(blast=/home/clgeng/software/blast/bin/psiblast,
                         database=/data/lixue/DBs/blast_dbs/nr_v20180204/nr)
        >>>
        >>> # generates the FASTA query
        >>> gen.get_fasta()
        >>>
        >>> # generates the PSSM
        >>> gen.get_pssm()
        >>>
        >>> # map the pssm to the pdb
        >>> gen.map_pssm()

        """

        self.caseID = caseID
        self.pdbdir = pdbdir
        self.pdbs = os.listdir(os.path.join(self.caseID,self.pdbdir))
        self.pdbs = list(filter(lambda x: x.endswith('.pdb'),self.pdbs))

        self.One2ThreeDict = {
        'A' : 'ALA', 'R' : 'ARG', 'N' : 'ASN', 'D' : 'ASP', 'C' : 'CYS', 'E' : 'GLU', 'Q' : 'GLN',
        'G' : 'GLY', 'H' : 'HIS', 'I' : 'ILE', 'L' : 'LEU', 'K' : 'LYS', 'M' : 'MET', 'F' : 'PHE',
        'P' : 'PRO', 'S' : 'SER', 'T' : 'THR', 'W' : 'TRP', 'Y' : 'TYR', 'V' : 'VAL',
        'B' : 'ASX', 'U' : 'SEC', 'Z' : 'GLX'
        }

        self.Three2OneDict = {v: k for k, v in self.One2ThreeDict.items()}

        self.psiblast_parameter = {
        0 : { 'wordSize':2, 'gapOpen':9,  'gapExtend':1, 'scoringMatrix':'PAM30' },
        1 : { 'wordSize':3, 'gapOpen':9,  'gapExtend':1, 'scoringMatrix':'PAM30' },
        2 : { 'wordSize':3, 'gapOpen':10, 'gapExtend':1, 'scoringMatrix':'PAM70' },
        3 : { 'wordSize':3, 'gapOpen':10, 'gapExtend':1, 'scoringMatrix':'BLOSUM80'},
        4 : { 'wordSize':3, 'gapOpen':11, 'gapExtend':1, 'scoringMatrix':'BLOSUM62'}
        }


    def get_fasta(self,chain=['A','B'],outdir='./fasta/'):

        """Extract the sequence of the chains and writes a fasta query file for each.

        Args:
            chain (list, optional): Name of the chains in the pdbs
            outdir (str, optional): name pf the output directory where to store the fast queries
        """

        outdir = os.path.join(self.caseID,outdir)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        pdb = os.path.join(os.path.join(self.caseID,self.pdbdir),self.pdbs[0])
        sqldb = pdb2sql(pdb)
        for c in chain:

            # get the unique residues
            res = sqldb.get_residues(chainID=c)

            # get the one letter resiude
            seq = ''
            count = 0
            for r in res:
                seq += self.Three2OneDict[r[1]]
                count += 1
                if count == 79:
                    seq += '\n'
                    count = 0

            # write the file
            fname = os.path.join(outdir,self.caseID + '_%s' %c + '.fasta')
            f = open(fname,'w')
            f.write('>%s' %self.caseID + '_%s\n' %c)
            f.write(seq)
            f.close()
        sqldb.close()


    def configure(self,blast=None,database=None):
        """Configure the blast executable and database.

        Args:
            blast (string) : Path to the psiblast executable
            database (string) : Path to the Blast database
        """

        self.blast = blast
        self.db = database

    def get_pssm(self,fasta_dir='fasta/',
                 outdir='pssm_raw/',
                 num_iterations=3,
                 run=True):

        """Compute the PSSM files

        Args:
            fasta_dir (str, optional): irectory where the fasta queries are stored
            blast (str, optional): path to the psiblast executable
            db (str, optional): path to the blast database
            outdir (str, optional): output directory where to store the pssm files
            num_iterations (int, optional): number of iterations for the blast calculations
        """

        fasta_dir = os.path.join(self.caseID,fasta_dir)
        outdir = os.path.join(self.caseID,outdir)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        out_fmt = '7 qseqid qgi qacc qaccver qlen sseqid sallseqid sgi sallgi   ,\
                   sacc saccver sallacc slen qstart qend sstart send qseq sseq  ,\
                   evalue bitscore score length pident nident mismatch positive ,\
                   gapopen gaps ppos frames qframe sframe btop staxids stitle   ,\
                   salltitles sstrand qcovs qcovhsp qcovus'

        for q in os.listdir(fasta_dir):

            # get the fasta quey
            query = os.path.join(fasta_dir,q)
            name = os.path.splitext(os.path.basename(query))[0]

            # set up the output names
            out_ascii_pssm = os.path.join(outdir,name + '.pssm')
            out_pssm = os.path.join(outdir,name + '.cptpssm')
            out_xml = os.path.join(outdir,name + '.xml')

            # get the parameters
            blast_param = self._get_psiblast_parameters(query)

            # set up the psiblast calculation
            psi_cline = NcbipsiblastCommandline(
                               self.blast,
                               db = self.db,
                               query = query,
                               evalue = 0.0001,
                               word_size = blast_param['wordSize'],
                               gapopen = blast_param['gapOpen'],
                               gapextend = blast_param['gapExtend'],
                               matrix = blast_param['scoringMatrix'],
                               outfmt = 7, #out_fmt,
                               comp_based_stats = 'T',
                               max_target_seqs = 2000,
                               save_each_pssm=True,
                               num_iterations=num_iterations,
                               save_pssm_after_last_round=True,
                               out_ascii_pssm = out_ascii_pssm,
                               out_pssm = out_pssm,
                               out = out_xml
                               )

            # check that it's correct
            psi_cline._validate()

            if run:

                # run the blast query
                psi_cline()

                # copyt the final pssm to its final name
                shutil.copy2(out_ascii_pssm + '.%d' %num_iterations, out_ascii_pssm)

                # remove all the other files
                for filename in glob.glob(out_pssm+'.*'):
                    os.remove(filename)
                for filename in glob.glob(out_ascii_pssm+'.*'):
                    os.remove(filename)
                os.remove(out_xml)


    def _get_psiblast_parameters(self,fasta_query):

        f = open(fasta_query)
        data =f.readlines()
        f.close()

        seq = 0
        for l in data[1:]:
            seq += len(l)

        if seq < 30:
            return self.psiblast_parameter[0]
        elif seq < 35:
            return self.psiblast_parameter[1]
        elif seq < 50:
            return self.psiblast_parameter[2]
        elif seq < 85:
            return self.psiblast_parameter[3]
        else:
            return self.psiblast_parameter[4]

    def map_pssm(self,pssm_dir='pssm_raw',outdir='pssm',chain=['A','B']):

        """Map the raw pssm files to the pdb files of the decoys

        Args:
            pssm_dir (str, optional): name pf the directory where the pssm are stored
            outdir (str, optional): name where thmapped pssm files are stored
            chain (list, optional): name of the chains
        """
        pssm_dir = os.path.join(self.caseID,pssm_dir)
        outdir = os.path.join(self.caseID,outdir)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        pf = os.listdir(pssm_dir)
        pssm_files = {}
        for c in chain:
            pssm_files[c] = list(filter(lambda x: x.endswith(c+'.pssm'),pf))[0]

        for p in self.pdbs:
            pdb = os.path.join(os.path.join(self.caseID,self.pdbdir),p)
            for c in chain:
                pssm = os.path.join(pssm_dir,pssm_files[c])
                write_mapped_pssm_pdb(pssm, pdb, c, outdir)

