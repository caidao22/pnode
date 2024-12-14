import csv
import numpy as np
class Recorder:
    def __init__(self):
        self.store = []
        self.current = dict()

    def __setitem__(self, key, value):
        for method in ['detach', 'cpu', 'numpy']:
            if hasattr(value, method):
                value = getattr(value, method)()
        if key in self.current:
            self.current[key].append(value)
        else:
            self.current[key] = [value]

    def capture(self, verbose=False):
        for i in self.current:
            if not isinstance(self.current[i],str):
                self.current[i] = np.mean(self.current[i])
        self.store.append(self.current.copy())
        self.current = dict()
        if verbose:
            for i in self.store[-1]:
                print('{}: {}'.format(i, self.store[-1][i]))
        return self.store[-1]

    def tolist(self):
        labels = set()
        labels = sorted(labels.union(*self.store))
        outlist = []
        for obs in self.store:
            outlist.append([obs.get(i, np.nan) for i in labels])
        return labels, outlist

    def writecsv(self, writer):
        
        labels, outlist = self.tolist()
        if isinstance(writer, str):
            outfile = open(writer, 'w')
            csvwriter = csv.writer(outfile)
            csvwriter.writerow(labels)
            csvwriter.writerows(outlist)
            outfile.close()
        else:
            csvwriter = writer
            csvwriter.writerow(labels)
            csvwriter.writerows(outlist)