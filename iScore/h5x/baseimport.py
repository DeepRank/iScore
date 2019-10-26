#%matplotlib inline
import numpy as np
import os
import subprocess as sp

from pdb2sql import pdb2sql


def launchPyMol(grp):

    cwd = os.getcwd() + '/'

    export_path =  cwd + grp.name + '/'
    if not os.path.isdir(export_path):
        os.mkdir(export_path)

    exec_fname = 'loadData.py'

    fname = export_path + exec_fname
    f = open(fname,'w')
    f.write('# can be executed with pymol -qRr loadData.py\n\n')
    f.write('import os\n')
    f.write('import pymol\n')
    f.write('from pymol.cgo import *\n')
    f.write('pymol.finish_launching()\n\n')

    f.write("# load the molecule\n")
    f.write("pymol.cmd.load('%s','complex')\n" %grp.attrs['pdbfile'])
    f.write("pymol.util.cbc(selection='(all)',first_color=7,quiet=1,legacy=0,_self=pymol.cmd)\n")

    db = pdb2sql(grp.attrs['pdbfile'])
    contact = []
    tmp_file = export_path + 'contact_res.pdb'

    for n in grp['nodes']:
        db.exportpdb(tmp_file,append=True,
                             chainID=n[0].decode('utf-8'),
                             resSeq= n[1].decode('utf-8'),
                             resName=n[2].decode('utf-8'))

    f.write("# load the contact\n")
    f.write("pymol.cmd.load('%s','contacts')\n" %tmp_file   )
    f.write("pymol.util.cbc(selection='(all)',first_color=7,quiet=1,legacy=0,_self=pymol.cmd)\n")
    f.write("pymol.cmd.show('stick','complex')\n\n")

    nodes  = []
    f.write('graph = [\n')
    for n in grp['nodes']:
        xyz = db.get('x,y,z',chainID=n[0].decode('utf-8'),
                             resSeq= n[1].decode('utf-8'),
                             resName=n[2].decode('utf-8'))
        xyz = np.mean(xyz,0)
        nodes.append(xyz)

        # if n[0] == b'A':
        #     f.write('COLOR, 0, 1, 0, \n')
        # else:
        #     f.write('COLOR, 0, 1, 1, \n')
        # f.write('SPHERE, %1.4f, %1.4f, %1.4f, 1.0, \n' %(xyz[0],xyz[1],xyz[2]))


    f.write('\nBEGIN, LINES,\n')
    f.write('COLOR, 1.0, 1.0, 1.0, \n')
    for e in grp['edges']:
        n1,n2 = nodes[e[0]],nodes[e[1]]
        f.write('VERTEX, %1.4f, %1.4f, %1.4f, \n' %(n1[0],n1[1],n1[2]))
        f.write('VERTEX, %1.4f, %1.4f, %1.4f, \n\n' %(n2[0],n2[1],n2[2]))
    f.write('END\n')
    f.write("]\npymol.cmd.load_cgo(graph,'graph', 1)\n\n")

    f.write("pymol.cmd.enable('complex')\n\n")

    f.close()

    sp.Popen('pymol -qQr ' + exec_fname, cwd = export_path,shell = True)