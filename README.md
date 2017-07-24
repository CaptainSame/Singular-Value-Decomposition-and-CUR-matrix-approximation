# Singular-Value-Decomposition-and-CUR-matrix-approximation

**SVD vs CUR Decomposition** Created by : Sameer Sharma

Python 3 should be installed.
Numpy package should be present.
sklearn package should be present.

The program svd_cur.py needs the name of the file containing the rating values as command line argument. 
The file should be kept in the same directory.

To compile and run, simply type python svd_cur.py in python command line followed by the name of the file to read.

The following two functions have been created:

1. mysvd: INPUT: takes the user-rating matrix and desired rank as input.
		  OUTPUT: returns the three matrices U, sigma and V_t resulting from the svd decomposition of the matrix.					
2. mycur: INPUT: takes the user-rating matrix and desired rank as input.
		  OUTPUT: returns the three matrices C, U and R resulting from the CUR decomposition of the matrix.

The program will output the following:
1. The dimensions of user-rating matrix.
2. The user-rating matrix.
3. SVD reconstruction error matrix followed by the time taken by SVD.
4. CUR reconstruction error matrix followed by the time taken by CUR.
