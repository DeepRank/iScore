import sys, os
import numpy as np
import matplotlib.pyplot as plt

class DataProcess(object):

    def __init__(self,fname):
        """Plot Class for iScore output.

        Example:

        >>> from iScore.process import DataProcess
        >>> proc = DataProcess('iScorePredict.dat')
        >>> proc.hit_rate()

        Args:
            fname(str): Name of the output file of iScore.predict
        """

        self.fname = fname
        self.data = self._read_data(fname)

    @staticmethod
    def _read_data(fname):
        """Read the data stored in the output file of iScore.predict.

        Args:
            fname(str): Name of the output file of iSCore.predict
        """

        if not os.path.isfile(fname):
            raise FileNotFoundError('File %s was not found' %fname)

        with open(fname,'r') as f:
            raw_data = f.readlines()

        data = dict()
        data['ID'], data['label'], data['pred'], data['val'] = [], [], [], []
        for l in raw_data[1:]:
            l = l.split()
            data['ID'].append(l[0])
            data['label'].append(l[1])
            data['pred'].append(l[2])
            data['val'].append(float(l[3]))

        return data

    def add_label(self,fname):
        """Add label to an existing data.

        Args:
            fname(str): file name containing the label format : ID label
        """

        if not os.path.isfile(fname):
            raise FileNotFoundError('File %s was not found' %fname)

        with open(fname,'r') as f:
            d = f.readlines()

        for l in d:
            ls = l.split()
            ID, label = ls[0], int(ls[1])

            try:
                index = self.data['ID'].index(ID)
                self.data['label'][index] = label

            except:
                raise ValueError("ID %s not found in data" %ID)


    def hit_rate(self,showfig=True, color='blue', legend="", figname=None):
        """Plot the hit rate of the prediction.

        Args:
            showfig (bool): show the figure or not
            color(str): color of the line
            legend(str): legend of the plot
            figname(str): name of the figure
        """

        indexsort = np.argsort(np.array(self.data["val"]))

        if 'None' in self.data['label']:
            raise ValueError('Some labels are not defined')
        else:
            self.data["label"] = np.array(self.data["label"]).astype('int')

        label = np.array(self.data["label"])[indexsort][::-1]
        print(label)
        hit = np.cumsum(label)
        fig,_ = plt.subplots()
        plt.plot(hit,c=color,label=legend)

        if figname is not None:
            fig.savefig(figname)
            plt.close()
        elif showfig:
            plt.show()




