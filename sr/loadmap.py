import yaml
import numpy as np
from sklearn.neighbors import KDTree


class ColorMap:
    def __init__(self, cfile, startpoint, endpoint):
        with open(cfile) as file:
            self.clist = yaml.load(file, Loader=yaml.FullLoader)

        # xdiff = [
        #     (
        #         self.clist[n][0] - self.clist[n - 1][0],
        #         self.clist[n][1] - self.clist[n - 1][1],
        #         self.clist[n][2] - self.clist[n - 1][2],
        #     )
        #     for n in range(1, len(self.clist))
        # ]
        # print(xdiff)
        # deltas = np.linspace(0, 1.0, 4, endpoint=False)[1:]
        # for el in xdiff:
        #     for d in deltas:
        #         print(d*el)


        self.N = len(self.clist) - 1
        self.start = startpoint
        self.end = endpoint
        assert len(self.clist) == len(
            set(self.clist)
        ), "The colors in this colormap are not unique!"

        # Convert to a more compact representation
        M = np.array(self.clist,dtype=np.uint8)
        self.kdt = KDTree(M, leaf_size=16, metric='euclidean')
        self.clist = M
        
        print("Total number of colors loaded for", cfile, "=", self.N+1)

    def convert(self, fnumber):
        """
        Returns colors. Expects input to be from self.start to self.end
        """
        #assert 0.0 <= fnumber <= 1.0
        
        return self.clist[int(self.N * (fnumber - self.start)/(self.end - self.start))]
        #index = int(self.N * fnumber)
        #return self.clist[index]

    def deconvert(self, rgbcolor):
        index = self.kdt.query([rgbcolor], k=1, return_distance=False)[0][0]
        # print("N = ", self.N+1)
        # print("Fraction = ", index/(self.N+1))
        # print("Index = ", index)
        return self.start + (index/float(self.N))*(self.end - self.start)
        #return (self.clist[index,:])

'''
if __name__ == "__main__":
    lab = ColorMap("lab.yaml", -1.0 , 1.0)
    print("0 maps to ", lab.convert(-1.0))
    print("1 maps to ", lab.convert(1.0))
    print("To and from from -1.0 to color :",  lab.deconvert(lab.convert(-1.0)))
    print("To and from from  1.0 to color :",  lab.deconvert(lab.convert(1.0)))
'''
