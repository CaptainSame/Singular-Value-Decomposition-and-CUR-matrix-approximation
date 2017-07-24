import sys
import numpy as np
import time
from collections import Counter
import random


def mysvd(matrix, rank):

	"""
	INPUT: matrix: user-rating matrix, rank: desired rank.
	OUTPUT: returns U, sigma and V_t resulting from the svd decomposition of the matrix.
	"""
	m = matrix.shape[0]
	n = matrix.shape[1]
	
	if((rank>m) or (rank>n)):
		print("error: rank greater than matrix dimensions.\n")
		return;
		
	matrix_t = matrix.T
	
	A = np.dot(matrix, matrix_t)						#calculate matrix multiplied by its transpose
	values1, v1 = np.linalg.eigh(A)						#get eigenvalues and eigenvectors
	v1_t = v1.T
	v1_t[values1<0] = 0						#discarding negative eigenvalues and corresponding eigenvectors(they are anyway tending to zero)
	v1 = v1_t.T
	values1[values1<0] = 0
	#values1 = np.absolute(values1)
		
	values1 = np.sqrt(values1)						#finding singular values.
	
	idx = np.argsort(values1)						#sort eigenvalues and eigenvectors in decreasing order
	idx = idx[: :-1]
	values1 = values1[idx]
	v1 = v1[:, idx]
	
	U = v1
	
	A = np.dot(matrix_t, matrix)						#calculate matrix transpose multiplied by matrix.
	values2, v2 = np.linalg.eigh(A)						#get eigenvalues and eigenvectors
	#values2 = np.absolute(values2)
	v2_t = v2.T
	v2_t[values2<0] = 0						#discarding negative eigenvalues and corresponding eigenvectors(they are anyway tending to zero)
	v2 = v2_t.T
	values2[values2<0] = 0
	
	values2 = np.sqrt(values2)						#finding singular values.
	
	idx = np.argsort(values2)						#sort eigenvalues and eigenvectors in decreasing order.
	idx = idx[: :-1]
	values2 = values2[idx]
	v2 = v2[:, idx]
	
	V = v2
	V_t = V.T										#taking V transpose.
	
	sigma = np.zeros((m,n))
	
	if(m>n):										#setting the dimensions of sigma matrix.
		
		sigma[:n, :] = np.diag(values2)
			
	elif(n>m):
		sigma[:, :m] = np.diag(values1)
		
	else:
		sigma[:, :] = np.diag(values1)
			
	if(m > rank):									#slicing the matrices according to the rank value.
		U = U[:, :rank]
		sigma = sigma[:rank, :]
	
	if(n > rank):
		V_t = V_t[:rank, :]
		sigma = sigma[:, :rank]
	
	check = np.dot(matrix, V_t.T)					
	#case = np.divide(check, values2[:rank])
	
	s1 = np.sign(check)
	s2 = np.sign(U)
	c = (s1==s2)
	
	for i in range(U.shape[1]):						#choosing the correct signs of eigenvectors in the U matrix.
		if(c[0, i]==False):
			U[:, i] = U[:, i]*-1
	
	
	return U, sigma, V_t

	

def mycur(matrix, rank):
	
	"""
	INPUT: matrix: user-rating matrix, rank: desired rank.
	OUTPUT: returns C, U and R resulting from the CUR decomposition of the matrix.
	"""
	m = matrix.shape[0]
	n = matrix.shape[1]
	
	if((rank>m) or (rank>n)):
		print("error: rank greater than matrix dimensions.\n")
		return;
		
	C = np.zeros((m, rank))
	R = np.zeros((rank, n))
	
	matrix_sq = matrix**2
	sum_sq = np.sum(matrix_sq)
	
	frob_col = np.sum(matrix_sq, axis=0)
	probs_col = frob_col/sum_sq				#probability of each column.
	
	count=0
	count1=0
	temp = 0
	idx = np.arange(n)						#array of column indexes.
	taken_c = []
	dup_c = []
	
	while(count<rank):
		i = np.random.choice(idx, p = probs_col)	#choosing column index based on probability.
		count1 = count1+1
		if(i not in taken_c):
			C[:, count] = matrix[:, i]/np.sqrt(rank*probs_col[i])	#taking column after dividing it with root of rank*probability.
			count = count+1
			taken_c.append(i)
			dup_c.append(1)
		else:										#discarding the duplicate column and increasing its count of duplicates.
			temp = taken_c.index(i)
			dup_c[temp] = dup_c[temp]+1
			
	C = np.multiply(C, np.sqrt(dup_c))				#multiply columns by root of number of duplicates.
			
	frob_row = np.sum(matrix_sq, axis=1)
	probs_row = frob_row/sum_sq					#probability of each row.
	
	count=0
	count1=0
	idx = np.arange(m)							#array of row indexes.
	taken_r = []
	dup_r = []
	
	while(count<rank):
		i = np.random.choice(idx, p = probs_row)			#choosing row index based on probability.
		count1 = count1+1
		if(i not in taken_r):
			R[count, :] = matrix[i, :]/np.sqrt(rank*probs_row[i])		#taking row after dividing it with root of rank*probability.
			count = count+1
			taken_r.append(i)
			dup_r.append(1)
		else:
			temp = taken_r.index(i)							#discarding the duplicate row and increasing its count of duplicates.
			dup_r[temp] = dup_r[temp]+1
		
	R = np.multiply(R.T, np.sqrt(dup_r))				#multiply rows by root of number of duplicates.
	R = R.T
	
	W = np.zeros((rank, rank))
	
	for i, I in enumerate(taken_r):
		for j, J in enumerate(taken_c):				#forming the intersection matrix W.
			W[i, j] = matrix[I, J]
	
	X, sigma, Y_t = mysvd(W,rank)					#svd decomposition of W.
	
	for i in range(rank):
		if(sigma[i,i] >= 1):						#taking pseudo-inverse of sigma.
			sigma[i,i] = 1/sigma[i,i]
		else:
			sigma[i,i] = 0
	
	U = np.dot(Y_t.T, np.dot(np.dot(sigma,sigma), X.T))		#finding U.
	
	return C, U, R
	

if __name__=='__main__':		

	np.random.seed(0)
	#filename = sys.argv[1]
	filename = "ratings.txt"
	list = [line.split(' ') for line in open(filename)]

	from sklearn import cross_validation as cv
	train_data, test_data = cv.train_test_split(list, test_size=0)

	num_users = len(Counter(elem[0] for elem in list))
	num_movies = len(Counter(elem[1] for elem in list))

	rating = np.zeros((num_users, num_movies))

	for elem in train_data:
		
		rating[int(elem[0])-1, int(elem[1])-1] = float(elem[2])
	
	print(num_users, num_movies)
	
	ranks = [50, 200, 500, 800, 1000]
	
	
	print("rating:", rating, "\n")
	
	for rank in ranks:
		print("\t***RANK :", rank, "***\n")
		t0 = time.time()
		u1, sigma1, vt1 = mysvd(rating, rank)
		t1 = time.time()
		
		d1 = np.dot(u1, np.dot(sigma1, vt1))
		
		diff = rating - d1
		print("svd error:", np.sum(diff**2))
		print("\nsvd time:", t1-t0, " secs")
		
		t2 = time.time()
		C, U, R = mycur(rating, rank)
		t3 = time.time()
		
		ans = np.dot(C, np.dot(U, R))
		print("\n\nCUR error:", np.sum((ans - rating)**2))
		print("\nCUR time:", t3-t2, " secs\n")
	