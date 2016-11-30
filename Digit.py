from math import *
import numpy as np
#import matplotlib.pyplot as plt
#import pdb
#from matplotlib import colors, ticker, cm

SIZE=28
TRAIN_SIZE=5000
TEST_SIZE=1000
k=0.1

def paser_files(fileimage,filelabel):
	file1=open(fileimage, 'r')
	file2=open(filelabel,'r')
	images_vec=[]
	labels_vec=[]
	image=[]
	temp=0
	for line in file1:
		image.append(line)
		temp+=1
		if temp>(SIZE-1):
			temp=0
			images_vec.append(image)
			image=[]

	for line in file2:
		labels_vec.append(line)

	return images_vec,labels_vec

def print_img(img_vec):
	for line in img_vec:
		for char in line:
			if char == '\n':
				print('')
			else:
				print(char, end="")

class ProbClass:
	def __init__(self):
		self.prob0=[]
		self.prob1=[]
		self.features=[]
		self.max_val=0
		self.min_val=0
		self.count=0 # counts how many of each number there are, adds up to 5000 before smoothing
		for x in range(SIZE*SIZE):
			self.prob0.append(0)
			self.prob1.append(0)
			self.features.append(0)

	def update_features(self,value,position):
		self.features[position[0]+position[1]*SIZE]+=value
# y is row, x is col
	def log_prob(self,value):
		for y in range(SIZE):
			for x in range(SIZE):
				self.prob1[x+y*SIZE]=np.log(float(self.features[x+y*SIZE])/self.count)
				self.prob0[x+y*SIZE]=np.log10(1-float(self.features[x+y*SIZE])/self.count)
#combine map and res
	def ret_res_map(self,feature_vec):
		temp=np.log(float(self.count)/TRAIN_SIZE) #same a P(class)
		for y in range(SIZE):
			for x in range(SIZE):
				if feature_vec[x+y*SIZE]==0:
					temp+=self.prob0[x+y*SIZE]
				else:
					temp+=self.prob1[x+y*SIZE]
		return temp

##training
images_vec,labels_vec=paser_files('trainingimages','traininglabels')
#print(len(images_vec),len(labels_vec))

train_numbers=[]
array_labels=[]
for i in range(10):
	train_numbers.append(ProbClass())
	array_labels.append(0)

for i in range(TRAIN_SIZE):
	temp_val=int(labels_vec[i])
	temp_img_val=images_vec[i]
	train_numbers[temp_val].count+=1#append to count 
	for y in range(SIZE):
			for x in range(SIZE):
				position=(x,y)
				if temp_img_val[y][x]=='':
					train_numbers[temp_val].update_features(0,position)
				if temp_img_val[y][x]=='#' or temp_img_val[y][x]=='+':
					train_numbers[temp_val].update_features(1,position)
for i in range(10):
	train_numbers[i].count+=k*2
	for y in range(SIZE):
			for x in range(SIZE):
				train_numbers[i].update_features(k,(x,y))
	train_numbers[i].log_prob(TRAIN_SIZE)

test_images_vec,test_labels_vec=paser_files('testimages','testlabels')
#print(len(test_images_vec),len(test_labels_vec))
probability=0
conf_mat=np.zeros((10,10))
map_val=[0]*10
classification=[0]*10
total=[0]*10
max_val=[-300]*10 #max MAP values in the test cases
min_val=[0]*10 #min MAP values in the test cases
pos_max=[0]*10 #positoin of those max values in the testimages
pos_min=[0]*10 #position of the min values in the testimages
#map_val=[0 for i in range(10)]
for i in range(TEST_SIZE):
	temp_val=int(test_labels_vec[i])
	temp_img_val=test_images_vec[i]
	total[temp_val]+=1
	test_numbers=ProbClass()
	test_numbers.count+=1
	for y in range(SIZE):
			for x in range(SIZE):
				position=(x,y)
				if temp_img_val[y][x]=='':
					test_numbers.update_features(0,position)
				if temp_img_val[y][x]=='#' or temp_img_val[y][x]=='+':
					test_numbers.update_features(1,position)

	for x in range(10):
		map_val[x]=train_numbers[x].ret_res_map(test_numbers.features)
		if(x==temp_val):
			#print(x,map_val[x],i)
			if(max_val[x]<map_val[x]):
				max_val[x]=map_val[x]
				pos_max[x]=i
			if(min_val[x]>map_val[x]):
				min_val[x]=map_val[x]
				pos_min[x]=i

	res_val=map_val.index(max(map_val))
	if(res_val==temp_val):
		probability+=1.0
		classification[temp_val]+=1.0
	conf_mat[res_val,temp_val]+=1

probability=probability/1000
sum_of_col=np.sum(conf_mat,axis=0)
for i in range(10):
	conf_mat[:,i]=conf_mat[:,i]/sum_of_col[i]
class_per_digit=np.divide(classification,total)
print("Accuracy : ", probability*100)
print((conf_mat))
print("Classificatoin per digit :" ,class_per_digit)
print(pos_max) #the position of the max likelihood digit in testimages
print(pos_min) #position of the min likelihood digit in the testimages

#prints all the test examples for hoghest and lowest posterior probabilities.
for i in range(10):
	print_img(test_images_vec[pos_max[i]]) 
	print_img(test_images_vec[pos_min[i]])


# because we have stored the probabilities in logarithmic form we can subtract in the odd ratios
def odd_ratio(num1,num2):
	prob1_num1=np.array(train_numbers[num1].prob1)#try reversin idk
	prob1_num1.shape=(SIZE,SIZE)
	prob1_num2=np.array(train_numbers[num2].prob1)
	prob1_num2.shape=(SIZE,SIZE)
	for i in range(SIZE):
		prob1_num1[i]=prob1_num1[i]#try reversing idk
		prob1_num2[i]=prob1_num2[i]
	odd_ratio_val=prob1_num1-prob1_num2
	return odd_ratio_val, prob1_num1, prob1_num2

'''def plot_func(num1,num2):
	array_odd, array1, array2=odd_ratio(num1,num2)
	plt.figure()
	plt.imshow(array1, cmap='plasma', interpolation='nearest')
	plt.colorbar()
	plt.figure()
	plt.imshow(array2, cmap='plasma', interpolation='nearest')
	plt.colorbar()
	plt.figure()
	plt.imshow(array_odd, cmap='plasma', interpolation='nearest')
	plt.colorbar()
plot_func(3,5) 
plot_func(8,5) 
plot_func(3,8) 
plot_func(9,7)'''
