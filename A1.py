import pandas as pd
import time
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
import csv
import math

#first change

#second change

# global variables
reviews = [] #store the data read from Jason file
k = 6 # shingle length
shingle_dic = {} # map shingles strings to number
shingle_index_counter = 0 # counter used in iteration
doc_index_counter = 0 # counter used in iteration
shinglelist = [] # list that store all the shingles 
doclist = [] # list that store all the doc numbers 
start_pointer = 0 # pointer
matrix = {} # matrix that store the doc index and it's corresponding shingle index
index_counter = 0 # counter used in iteration 
index_doc_dic = {} #given doc number find index number 
doc_index_dic = {} #given index number find doc number
buck_list = [] #store the similar pairs
m = 100 
rows = 20
band = 5
p = 0
# Function used by problem 1
# define a function to join the strings in a list
def joinfunction(review_str):
		return ' '.join(review_str)

#Function used by problem 2
#get the shinglelist and doclist for sparse matrix
def shinglefunction1(review_str):
    global k, shingle_dic, shingle_index_counter, shinglelist, start_pointer, doclist, doc_index_counter  
    for i in range(len(review_str)-k+1):
        if review_str[i:i+k] not in shingle_dic:
            shingle_dic['{}'.format(review_str)[i:i+k]] = shingle_index_counter
            shinglelist.append(shingle_index_counter)
            doclist.append(doc_index_counter)
            shingle_index_counter = shingle_index_counter + 1
        else:
            if shingle_dic['{}'.format(review_str)[i:i+k]] not in shinglelist[start_pointer:]:
                shinglelist.append(shingle_dic['{}'.format(review_str)[i:i+k]])
                doclist.append(doc_index_counter)
    start_pointer = len(shinglelist)
    doc_index_counter = doc_index_counter + 1

#Function used by problem 4
#A new way to store data in a dense matrix
def shinglefunction2(review_str):
	global k, shingle_dic, shingle_index_counter, doc_index_counter, matrix, index_counter, index_doc_dic, doc_index_dic
	if len(review_str) >= k:
		index_doc_dic[doc_index_counter] = index_counter
		doc_index_dic[index_counter] = doc_index_counter
		matrix[doc_index_counter] = []  
		for i in range(len(review_str)-k+1):
			if review_str[i:i+k] not in shingle_dic:
				shingle_dic['{}'.format(review_str)[i:i+k]] = shingle_index_counter
				matrix[doc_index_counter].append(shingle_index_counter)
				shingle_index_counter = shingle_index_counter + 1
			else:
				if shingle_dic['{}'.format(review_str)[i:i+k]] not in matrix[doc_index_counter]:
					matrix[doc_index_counter].append(shingle_dic['{}'.format(review_str)[i:i+k]])
		doc_index_counter = doc_index_counter + 1
	index_counter = index_counter + 1

#Function used by problem 3
#calculate jaccard distance
def jaccard_distance(list1, list2):
	intersection = len(list(set(list1).intersection(list2)))
	union = (len(list1) + len(list2)) - intersection
	return 1-(float(intersection) / union)

#Functions used by problem 5
#if a number is prime
def isPrime(n):  
	if(n <= 1): 
		return False
	if(n <= 3): 
		return True 
	if(n % 2 == 0 or n % 3 == 0): 
		return False
	for i in range(5,int(math.sqrt(n) + 1), 6):  
		if(n % i == 0 or n % (i + 2) == 0): 
			return False   
	return True

#find the next prime number
def nextPrime(N):  
	if (N <= 1): 
		return 2
	prime = N 
	found = False
	while(not found): 
		prime = prime + 1
		if(isPrime(prime) == True): 
			found = True
	return prime 

#Minhash function
def Minhash(shingle_index_list, a_array, b_array):
	shingle_index_array = np.array(shingle_index_list)
	sig_value_list = ((a_array * shingle_index_array + b_array)%p).min(1)
	return sig_value_list

#LSH function
def LSH(sig_rows):
	global rows, p
	a1 = []
	b1 = []
	for i in range(rows):
		a1.append(np.random.randint(low = 1, high = p, size = 1))
		b1.append(np.random.randint(low = 1, high = p, size = 1))
	a1_array = np.array(a1).reshape(rows, 1)
	b1_array = np.array(b1).reshape(rows, 1)
	sig_matrix_array = np.array(sig_rows)
	row_value = np.sum((a1_array*sig_matrix_array+b1_array)%p, axis=0)
	return row_value

#Functions for each problem:
def problem1():
	global reviews
	stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves",
				 "you", "your", "yours", "yourself", "yourselves", "he", "him",
				 "his", "himself", "she", "her", "hers", "herself", "it", "its",
				 "itself", "they", "them", "their", "theirs", "themselves", "what",
				 "which", "who", "whom", "this", "that", "these", "those", "am",
				 "is", "are", "was", "were", "be", "been", "being", "have", "has",
				 "had", "having", "do", "does", "did", "doing", "a", "an", "the",
				 "and", "but", "if", "or", "because", "as", "until", "while", "of",
				 "at", "by", "for", "with", "about", "against", "between", "into",
				 "through", "during", "before", "after", "above", "below", "to",
				 "from", "up", "down", "in", "out", "on", "off", "over", "under",
				 "again", "further", "then", "once", "here", "there", "when", "where",
				 "why", "how", "all", "any", "both", "each", "few", "more", "most",
				 "other", "some", "such", "no", "nor", "not", "only", "own", "same",
				 "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
				 "should", "now"]
	punctuation = "!#\"$%&'()*+,-./:;<=>?@[\]^_`{|}~."

	reviews = pd.read_json('/home/weichu/Desktop/Archive/amazonReviews.json', lines=True)
	reviews = reviews.drop(['asin', 'reviewerName', 'helpful', 'overall', 'summary', 'unixReviewTime', 'reviewTime'], axis = 1)
	# replace all punctuations with space
	for i in punctuation:
		reviews['reviewText'] = reviews['reviewText'].str.replace('{}'.format(i), ' ')
	# conver strings into lowercase and split sentences into words
	reviews['reviewText'] = reviews['reviewText'].str.lower().str.split()
	# delete all stopwords 
	reviews['reviewText'] = reviews['reviewText'].apply(lambda x: [item for item in x if item not in stopwords])
	# join the strings in a list
	reviews['reviewText'] = reviews['reviewText'].apply(joinfunction)


def problem2():
	global reviews, doclist, shinglelist
	reviews['reviewText'].apply(shinglefunction1)
	data = [1]*len(doclist)
	# create sparse matrix
	sparse_matrix = sparse.coo_matrix((data, (doclist, shinglelist)))

def problem4():
	global reviews, matrix, shingle_dic, shingle_index_counter, doc_index_counter
	shingle_dic = {}
	shingle_index_counter = 0
	doc_index_counter = 0
	reviews['reviewText'].apply(shinglefunction2)
	print("Total number of review is ", index_counter)
	print("number of valid review is ", len(matrix))
	print("number of shingle is ", len(shingle_dic))

def problem3():
	global matrix
	similarlist = [] # list to store Jaccard Distance between pairs
	for i in range(10000):
	    random = np.random.randint(low = 0, high = len(matrix)-1, size = 2)
	    similarlist.append(jaccard_distance(matrix[random[0]], matrix[random[1]]))
	print("Average Jaccard Distance among all pairs is ", sum(similarlist)/len(similarlist))
	print("Lowest Jaccard Distance among all pairs is ", min(similarlist))
	plt.hist(similarlist, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
	plt.xlabel('Jaccard Distance')
	plt.ylabel('Number of Documents')
	plt.title('Histogram of Jaccard Distance')
	plt.show()

def problem5():
	global p, buck_list, matrix
	p = nextPrime(len(shingle_dic))
	print("prime number p is ", p)
	# do Minhash:
	a = []
	b = []
	for i in range(m):
		a.append(np.random.randint(low = 1, high = p, size = 1))
		b.append(np.random.randint(low = 1, high = p, size = 1))
	a_array = np.array(a).reshape(m, 1)
	b_array = np.array(b).reshape(m, 1)
	sig_matrix = np.zeros([m, len(matrix)], np.int) # signature matrix
	for doc in range(len(matrix)):
		sig_matrix[:, doc] = Minhash(matrix[doc], a_array, b_array)
	print("After Minhash, sig_matrix is ready")
	#do LSH:
	sig_matrix_new = []# new signature matrix after combine the rows
	for i in range(band):
		row_value = LSH(sig_matrix[(i*rows):((i+1)*rows), :])
		sig_matrix_new.append(row_value)
	print("After LSH, sig_matrix_new is ready")
	# map the similar pairs to the same buck:
	sig_matrix_new = np.array(sig_matrix_new)
	for row in range(band):
		buck = {}
		for doc in range(len(matrix)):
			if sig_matrix_new[row,doc] not in buck:
				buck[sig_matrix_new[row,doc]] = []
				buck[sig_matrix_new[row,doc]].append(doc)
			else:
				buck[sig_matrix_new[row,doc]].append(doc)
		for key in list(buck):
			if len(buck[key])<=1:
				del buck[key] # delete the buck that is empty or only have one value
			else:
				if buck[key] not in buck_list:
					buck_list.append(buck[key])# put the same pairs into the list 
	print("buck_list is ready")
	with open("/home/weichu/Desktop/Archive/similar_pairs.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(buck_list)
	print("Write in CSV finished")

def problem6():
	global buck_list, reviews, matrix
	# create a search dic that for each document key, store the similar documents index in its value
	search_dic = {new_list: [] for new_list in range(len(matrix))}
	for i in range(len(buck_list)):
		for j in range(len(buck_list[i])):
			for n in range(len(buck_list[i])):
				if buck_list[i][n] != buck_list[i][j] and buck_list[i][n] not in search_dic[buck_list[i][j]]:
					search_dic[buck_list[i][j]].append(buck_list[i][n])
	print("search_dic ready")

	query = 'y'
	while query != 'n':
		print("******************************************************************")
		print("******************************************************************")
		try:
			query = input('Please enter the INDEX[0-157835] of the review that you want to query, quit with n: ')
			if int(query)>=0 and int(query)<=157835:
				query = int(query)
				print("The information of the review you want to query is shown as below:")
				print("ReviewerID:")
				print(reviews['reviewerID'][query])
				print("Review Text:")
				print(reviews['reviewText'][query])
				print("-----------------------------------------")
				if query in doc_index_dic:
					if (len(search_dic[doc_index_dic[query]]) == 0):
						print("There is no similar reviews")
					else:
						print("The nearest neighbors are as follow:")
						for doc in search_dic[doc_index_dic[query]]:
							print("ReviewerID:")
							print(reviews['reviewerID'][index_doc_dic[doc]])
							print("Review Text:")
							print(reviews['reviewText'][index_doc_dic[doc]])
							print("-----")
				else:
					print("The length of the review you want to query is too short, please find another review")
		except ValueError as e:
			if e != 'n':
				print("The INDEX you entered is invalid please enter again")
			else:
				print("Program End")


def main():
	#Problem 1
	print("Running Problem 1...")
	start_time = time.time()
	problem1()
	print("Document has been processed")
	print("Problem 1 Finished")
	print("--- %s seconds ---" % (time.time() - start_time))

	#Problem 2
	print("Running Problem 2...")
	middle_time = time.time()
	problem2()
	print("Sparse Matrix is ready")
	print("Problem 2 Finished")
	print("--- %s seconds ---" % (time.time() - middle_time))

	#Problem 4
	print("Running Problem 4...")
	middle_time = time.time()
	problem4()
	print("New matrix to store data is ready")
	print("Problem 4 Finished")
	print("--- %s seconds ---" % (time.time() - middle_time))

	#Problem 3
	print("Running Problem 3...")
	middle_time = time.time()
	problem3()
	print("Problem 3 Finished")
	print("--- %s seconds ---" % (time.time() - middle_time))

	#Problem 5
	print("Running Problem 5...")
	middle_time = time.time()
	problem5()
	print("Problem 5 Finished")
	print("--- %s seconds ---" % (time.time() - middle_time))

	#Problem 6
	print("Running Problem 6...")
	middle_time = time.time()
	problem6()
	print("Problem 6 Finished")
	print("--- %s seconds ---" % (time.time() - middle_time))
	print("total time")
	print("--- %s seconds ---" % (time.time() - start_time))
