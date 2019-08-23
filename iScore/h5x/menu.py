from h5xplorer.menu_tools import *
from h5xplorer.menu_plot import *

def context_menu(self, treeview, position):

    """Generate a right-click menu for the items"""

    all_item = get_current_item(self,treeview,single=False)

    if len(all_item) == 1:

        item = all_item[0]
        data = get_group_data(get_current_hdf5_group(self,item))

        if data is None:
            list_operations = ['Print attrs', 'PyMol']

        elif data.ndim == 1:
            list_operations = ['Print attrs','-','Plot Hist', 'Plot Line']

        elif data.ndim == 2:
            list_operations = ['Print attrs','-','Plot Hist', 'Plot Map']

        else:
            list_operations = ['Print attrs']

        action,actions = get_actions(treeview,position,list_operations)
        #action, actions = get_multilevel_actions(treeview,position,list_operations,list_sub)

        if action == actions['Print attrs']:
            send_dict_to_console(self,item,treeview)

        if 'Plot Hist' in actions:
            if action == actions['Plot Hist']:
                plot_histogram(self,item,treeview)

        if 'Plot Line' in actions:
            if action == actions['Plot Line']:
                plot_line(self,item,treeview)

        if 'Plot Map' in actions:
            if action == actions['Plot Map']:
                plot2d(self,item,treeview)

        if 'PyMol' in actions:
            if action == actions['PyMol']:

                grp = get_current_hdf5_group(self,item)
                data_dict = {'_grp':grp}
                treeview.emitDict.emit(data_dict)

                cmd = 'launchPyMol(_grp)'
                data_dict = {'exec_cmd':cmd}
                treeview.emitDict.emit(data_dict)