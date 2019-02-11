import numpy as np

hidden_num=1;
datatype=np.float32

mm=0.3
epoch_size=1
epoch_num=100
lr=0.5

inputs=np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]],dtype=datatype)
true_out=np.array(([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]))
print('inputs shape:\n',inputs.shape)
print('inputs:\n',inputs)
print('true_out shape:\n',true_out.shape)
print('true_out:\n',true_out)


#定义隐藏层函数
def hidden_layer(inputs,units,activ=None,train=0,wg=None,layer_order=None):
    #判断输入是否为二维数组
    #print(inputs.shape)
    assert len(inputs.shape)==2
    
    #初始或赋值权重矩阵
    if train==0:
        weights=np.array(np.random.randn(inputs.shape[1],units))
    else:
        weights=wg
    print('layer_%d-weights\n'%(layer_order),weights)
    
    #加权求和
    bias=np.zeros((inputs.shape[0],units))
    sum_weights=np.matrix(inputs)*np.matrix(weights)+bias
    print('layer_%d-sum_weights\n'%(layer_order),sum_weights)
    
    #送入激活函数
    if activ!=None: 
        result=activ(sum_weights)
    else:
        result=sum_weights
    print('layer_%d-result\n'%(layer_order),result)
    
    result=np.array(result)
    return result,weights,bias

#定义logist激活函数
def logist(x):
    y=1/(1+np.exp(-x))
    return y
#计算隐藏层误差函数


h1,weights1,bias1=hidden_layer(np.array([inputs[0]]),3,logist,layer_order=1)
output,weights_out,bias_out=hidden_layer(h1,2,logist,layer_order=2)

#计算输出层误差
E_out=output*(1-output)*(true_out[0]-output)
print('E_out:\n',E_out)

#计算隐藏层误差
E_h1=np.reshape(np.sum(weights_out*E_out,axis=1),(epoch_size,-1))
print('weights_out*E_out',E_h1)
print('h1*(1-h1)',h1*(1-h1))
E_h1=h1*(1-h1)*E_h1
print('E_h1:',E_h1)

#调整输出层参数
print('lr*np.multiply(output,E_out):',lr*np.multiply(h1.T,E_out))
weights_out=weights_out+lr*np.multiply(h1.T,E_out)
print('weights_out:',weights_out)
bias_out=weights_out+lr*E_out

#调整隐藏层参数
weights1=weights1+lr*np.multiply(np.array([inputs[0]]).T,E_h1)
print('weights1',weights1)
bias1=weights1+lr*E_h1
print('bias1:',bias1)

#赋值参数
sample=sample+1
h1,weights1,bias1=hidden_layer(np.array([inputs[0]]),3,logist,train=1,wg=weights1,layer_order=1)
output,weights_out,bias_out=hidden_layer(h1,2,logist,train=1,wg=weights_out,layer_order=2)
print('%d output:'%(sample),output)