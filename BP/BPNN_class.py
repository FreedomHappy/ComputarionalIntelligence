
# coding: utf-8

import random
import numpy as np
import time


def sigmoid(z):
	return 1/(1+np.exp(-z))
def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))




class BPNN(object):
	def __init__(self, sizes):
		self.error_change=0.0001
		self.sum_layers=len(sizes)
		self.sizes=sizes
		self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
		self.biases=[np.random.randn(y,1) for y in sizes[1:]]
		print('self.weights:\n',self.weights)
		print('self.biases:\n',self.biases)
		
		
	def feedforward(self,input):
		for w,b in zip(self.weights,self.biases):
			input=sigmoid(np.dot(w,input)+b)
		return input
		
	def test(self,test_data):#test_data无y值
		print('\n测试结果为：\n')
		for data in test_data:
			result=self.feedforward(data.reshape(-1,1))
			print('测试样本{0}结果为：{1}'.format(abs(data),abs(result)))
		pass
	
	def SGD(self,train_data,mini_batches_size,epochs,lr,test_data=None):
		n=len(train_data)
		for i in range(epochs):
			random.shuffle(train_data)
			for k in range(0,n,mini_batches_size):
				mini_batch=train_data[k:k+mini_batches_size]
				# print('mini_batch:',mini_batch)
				error=self.update_par(mini_batch,lr)
			# print('{0}_epochs complete'.format(i))
		self.train__information(epochs,error)
		
		if test_data.all():
			self.test(test_data)
		pass
		
	def update_par(self,mini_batch,lr):
		sum_delta_w=[np.zeros(w.shape) for w in self.weights]
		sum_delta_b=[np.zeros(b.shape) for b in self.biases]
		for x,y in mini_batch:
			delta_w,delta_b=self.back_propagation(x,y)
			sum_delta_w=[ sw+dw for sw,dw in zip(sum_delta_w,delta_w)]
			sum_delta_b=[ sb+db for sb,db in zip(sum_delta_b,delta_b)]
		
		if (abs(sum_delta_b[-1]/len(mini_batch)))<self.error_change: return sum_delta_b[-1]/len(mini_batch)
		
		self.weights = [w-(lr/len(mini_batch))*nw
						for w, nw in zip(self.weights, sum_delta_w)]
		self.biases = [b-(lr/len(mini_batch))*nb
						for b, nb in zip(self.biases, sum_delta_b)]
		# for w,b,sw,sb in zip(self.weights,self.biases,sum_delta_w,sum_delta_b):
			# w=w-(lr/len(mini_batch))*sw
			# b=b-(lr/len(mini_batch))*sb
			
		return sum_delta_b[-1]/len(mini_batch)
		
	def back_propagation(self,x,y):
		# print('x:',x)
		# print('y:',y)
		#feedforward
		zs=[]
		activations=[x]
		activation=x
		for w,b in zip(self.weights,self.biases):
			z=np.dot(w, activation)+b
			zs.append(z)
			activation=sigmoid(z)
			# print('activation:',activation)
			activations.append(activation)
		# print('y:',activation)
		#back_propagation
		delta_w=[np.zeros(w.shape) for w in self.weights]
		delta_b=[np.zeros(b.shape) for b in self.biases]
		#计算输出层改变率
		error=sigmoid_prime(zs[-1])*(activations[-1]-y)
		# print('error:',error)
		delta=np.dot(error,activations[-2].transpose())
		delta_b[-1]=error
		delta_w[-1]=delta
		#计算隐藏层改变率
		for layer in range(2,self.sum_layers):
			error=sigmoid_prime(zs[-layer])*np.dot(self.weights[-layer+1].transpose(),error)
			delta_b[-layer]=error
			delta=np.dot(error,activations[-layer-1].transpose())
			delta_w[-layer]=delta
		# print('error:',error)
		return delta_w,delta_b

	def train__information(self,epochs,error):
		print('\n训练结果：\n最终迭代次数：{0} 误差{1}'.format(abs(epochs),abs(error)))
		for i in range(self.sum_layers-1):
			print('\n{0} to {1}层连接权值为：\n'.format(i+1,i+2),self.weights[i])
		for i in range(self.sum_layers-1):
			print('\n{0} to {1}层阈值为：\n'.format(i+1,i+2),self.biases[i])
	
	
	
def parseData(xs,ys):
	xs=np.array(xs).reshape(-1,len(xs[0]),1)
	ys=np.array(ys).reshape(-1,len(ys[0]),1)
	train_data=[]
	for x,y in zip(xs,ys):
		train_data.append((x,y))
	return train_data
	
def main():
	#array问题
	input=np.array([[[[1],[1],[1]],[[0],[0],[0]]]])
	input2=np.array([[[[1],[1],[1]],[[0]]]])
	print(input[0])
	print(input2[0])
	
	xs=[[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0],[0.1 ,1.0]]#shape=sample_num*x_size
	ys=[[0.0],[1.0],[1.0],[0.0],[1.0]]		#shape=sample_num*y_size
	train_data=parseData(xs,ys)
	test_data=np.array([[0.05,0.1],[0.2,0.9],[0.86,0.95]])
	
	bp=BPNN([2,3,1])
	bp.SGD(train_data,1,3000,0.6,test_data=test_data)
	
	
	
	
if __name__=='__main__':
	main()
	