import os
import sys
import unittest
import numpy as np
from scipy import weave

sys.path.append(os.environ['GIT_REPO'] + '/source-code/cython-modules')
import c_oriented_histograms
import c_arrayloop

sys.path.append(os.environ['GIT_REPO'] + '/source-code/event-detection')
import _hog

class oriented_histograms(unittest.TestCase):
    def test_c_calculate_histograms(self):
        sx, sy = 10, 10
        cx, cy = 5, 5
        n_orientations = 2
        n_cellsx, n_cellsy = int(np.floor(sx // cx)), int(np.floor(sy // cy))

        mag = np.arange(sx*sy).reshape(sy, sx)
        ang = (mag * np.pi) % 180

        orientation_histogram_cython = np.zeros([n_cellsy, n_cellsx, n_orientations])
        c_oriented_histograms.calculate_histograms(mag.astype(np.double),
                                                   ang.astype(np.double),
                                                   cx, cy, sx, sy, n_cellsx, n_cellsy, n_orientations,
                                                   orientation_histogram_cython)

        orientation_histogram_weave = _hog.calculate_histograms(mag,
                                                                ang,
                                                                cx, cy, sx, sy, n_cellsx, n_cellsy, n_orientations)

        print orientation_histogram_cython
        print orientation_histogram_weave

        orientation_histogram_cython = orientation_histogram_cython.ravel()
        orientation_histogram_weave = orientation_histogram_weave.ravel()

        diff = orientation_histogram_cython - orientation_histogram_weave
        self.assertTrue(np.abs(diff).sum() <= 1E-8)

    def test_cython_weave(self):
        code = """
            int c = 0;
            int i, j, k;
            for (i = 0; i < NA[0]; i++)
                for (j = 0; j < NA[1]; j++)
                    A[i * NA[1] + j] = c++;
            for (i = 0; i < NB[0]; i++)
                for (j = 0; j < NB[1]; j++)
                    for (k = 0; k < NB[2]; k++)
                        B[(i * NB[1] + j) * NB[2] + k] = c++;

        """
        A = np.zeros([3,4])
        B = np.zeros([3,4,5])
        weave.inline(code, ['A', 'B']);
        A_weave = np.copy(A)
        B_weave = np.copy(B)

        A = np.zeros([3,4])
        B = np.zeros([3,4,5])
        c_arrayloop.array2d_3d_loop(A, B, np.array(A.shape), np.array(B.shape))

        self.assertTrue((A_weave == A).any())
        self.assertTrue((B_weave == B).any())

if __name__ == '__main__':
   unittest.main()

