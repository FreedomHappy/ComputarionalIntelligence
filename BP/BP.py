{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (<ipython-input-1-20029cf27209>, line 39)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"<ipython-input-1-20029cf27209>\"\u001b[1;36m, line \u001b[1;32m39\u001b[0m\n\u001b[1;33m    for i range(100):\u001b[0m\n\u001b[1;37m              ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "hidden_num=1;\n",
    "datatype=np.float32\n",
    "\n",
    "mm=0.3\n",
    "epoch_size=4\n",
    "epoch_num=100\n",
    "lr=0.5\n",
    "\n",
    "inputs=np.matrix([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]],dtype=datatype)\n",
    "true_out=np.matrix(np.reshape([0.0,1.0,1.0,0.0],(epoch_size,-1)))\n",
    "print(inputs.shape)\n",
    "print('true_out:',true_out)\n",
    "\n",
    "def hidden_layer(inputs,units,activ=None,train=0,wg=None):\n",
    "    if train==0:\n",
    "        weights=np.matrix(np.random.randn(inputs.shape[1],units))\n",
    "    else:\n",
    "        weights=wg\n",
    "    print(weights)\n",
    "    \n",
    "    bias=np.zeros(units)\n",
    "    res_weights=inputs*weights+bias\n",
    "    print(res_weights)\n",
    "    \n",
    "    if activ!=None: \n",
    "        result=activ(res_weights)\n",
    "    print(result)\n",
    "    \n",
    "    return result,weights,bias\n",
    "   \n",
    "def logist(x):\n",
    "    y=1/(1+np.exp(-x))\n",
    "    return y\n",
    "\n",
    "sample=0\n",
    "\n",
    "for i range(epoch_num):\n",
    "    sample=sample+1\n",
    "    h1,weights1,bias1=hidden_layer(inputs[sapmle%4],2,logist)\n",
    "    output,weights_out,bias_out=hidden_layer(h1,1,logist)\n",
    "    print('%d ouput:'%(sample),output)\n",
    "    #print(true_out-output)\n",
    "    #print(1-output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "    E_out=np.array(output)*np.array((1-output))*np.array((true_out[0]-output))\n",
    "    #print(E_out)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "    #print(h1)\n",
    "    #print(1-h1)\n",
    "    output_T=output.T\n",
    "   # print(output_T)\n",
    "  #  print('weights_out:',weights_out)\n",
    "  #  print(weights_out*output_T)\n",
    "    h1_entroy=np.array(h1)*np.array(1-h1)\n",
    "    E_h1=np.array(h1_entroy)*np.array(((weights_out*output_T).T))\n",
    "   # print(E_h1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "    weights_out=weights_out+lr*np.multiply(output,E_out)\n",
    "   # print(lr*np.multiply(output,E_out))\n",
    "   # print(weights_out)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "print(weights1)\n",
    "print(lr*np.multiply(h1,E_h1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "    bias_out=weights_out+lr*E_out"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "   # print(weights1) \n",
    "  #  print(lr*np.multiply(h1,E_h1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "    weights1=weights1+lr*np.multiply(h1,E_h1)\n",
    "   # print(weights1)\n",
    "\n",
    "    bias1=weights1+lr*E_h1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "    sample=sample+1\n",
    "    h1,weights1,bias1=hidden_layer(inputs[sample%4],2,logist,train=1,wg=weights1)\n",
    "    output,weights_out,bias_out=hidden_layer(h1,1,logist,train=1,wg=weights_out)\n",
    "    print('%d output:'%(sample),output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
