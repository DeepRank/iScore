#!/usr/bin/env python

from h5xplorer.h5xplorer import h5xplorer
import h5x_menu

baseimport = '/home/nico/Documents/projects/iScore/h5x/baseimport.py'
app = h5xplorer(h5x_menu.context_menu,baseimport=baseimport,extended_selection=False)