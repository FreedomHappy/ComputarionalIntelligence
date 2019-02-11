
# coding: utf-8

# In[1]:


import numpy as np

hidden_num=1;
datatype=np.float32

mm=0.3
epoch_size=4
epoch_num=1500
lr=1/epoch_num

inputs=np.matrix([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]],dtype=datatype)
true_out=np.matrix(np.reshape([0.0,1.0,1.0,0.0],(epoch_size,-1)))
print(inputs.shape)
print('true_out:',true_out)

def hidden_layer(inputs,units,activ=None,train=0,wg=None):
    if train==0:
        weights=np.matrix(np.random.randn(inputs.shape[1],units))
    else:
        weights=wg
    print(weights)
    
    bias=np.zeros(units)
    res_weights=inputs*weights+bias
    print(res_weights)
    
    if activ!=None: 
        result=activ(res_weights)
    print(result)
    
    return result,weights,bias
   
def logist(x):
    y=1/(1+np.exp(-x))
    return y

sample=0

for i in range(epoch_num):
    sample=sample+1
    h1,weights1,bias1=hidden_layer(inputs[sample%4],2,logist)
    output,weights_out,bias_out=hidden_layer(h1,1,logist)
    print('%d ouput:'%(sample),output)
    #print(true_out-output)
    #print(1-output)
    E_out=np.array(output)*np.array((1-output))*np.array((true_out[sample%4]-output))
    print('%d E_out:'%(sample),E_out)
    #print(h1)
    #print(1-h1)
    E_out_T=E_out.T
    # print(output_T)
    #  print('weights_out:',weights_out)
    #  print(weights_out*output_T)
    h1_entroy=np.array(h1)*np.array(1-h1)
    E_h1=np.array(h1_entroy)*np.array(((weights_out*E_out_T).T))
    # print(E_h1)
    weights_out=weights_out+lr*np.multiply(output,E_out)
    # print(lr*np.multiply(output,E_out))
    # print(weights_out)
    # print(weights1)
    # print(lr*np.multiply(h1,E_h1))
    bias_out=weights_out+lr*E_out
    # print(weights1) 
    # print(lr*np.multiply(h1,E_h1))

    # In[ ]:


    weights1=weights1+lr*np.multiply(h1,E_h1)
    # print(weights1)

    bias1=weights1+lr*E_h1

    # In[ ]:
    sample=sample+1
    h1,weights1,bias1=hidden_layer(inputs[sample%4],2,logist,train=1,wg=weights1)
    output,weights_out,bias_out=hidden_layer(h1,1,logist,train=1,wg=weights_out)
    print('%d output:'%(sample),output)

