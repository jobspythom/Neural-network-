#first Python neuralNetwork

import numpy
import scipy.special
#import matplotlib.pyplot




class neuralNetwork:
    #初始化神经网络，输入节点数量，隐藏层节点数量，输出层节点数量和学习率
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes

        #设置输入层到隐藏层，隐藏层到输出层的权重,权重采用均值为0，标准差为上节点数量平方根的倒数，也就是说节点数量越多，权重越接近0
        self.wih=numpy.random.normal(0.0,pow(self.inodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.onodes,self.hnodes))
        #初始化学习率，学习率介于0-1之间，准确性考虑学习率不应设置过大，面对学习数据较少并且世代数少的神经网络，设置太小也会影响神经网路的准确性
        self.lr=learningrate
        #激活函数，为1/(1+e**(-x))
        self.activation_function=lambda x:scipy.special.expit(x)

        pass



     #训练函数
    def train(self,input_lists,targets_lists):
         #接收训练数据喝目标数据，将其转至
         inputs=numpy.array(input_lists,ndmin=2).T
         targets=numpy.array(targets_lists,ndmin=2).T
         #计算第二层的输入
         hidden_input=numpy.dot(self.wih,inputs)
         hidden_output=self.activation_function(hidden_input)


         #计算输出结果
         final_input=numpy.dot(self.who,hidden_output)
         final_output=self.activation_function(final_input)

         #计算输出结果与标准值之间的误差
         output_errors=targets-final_output
         hidden_errors=numpy.dot(self.who.T,output_errors)
         #根据误差调整相应权重
         self.who+=self.lr*numpy.dot((output_errors*final_output*(1-final_output)),numpy.transpose(hidden_output))
         self.wih+=self.lr*numpy.dot((hidden_errors*hidden_output*(1-hidden_output)),numpy.transpose(inputs))

         pass



     #验证函数
    def query(self,input_list):
        inputs=numpy.array(input_list,ndmin=2).T
        #根据验证数据计算第一层结果
        hidden_input=numpy.dot(self.wih,inputs)
        hidden_outputs=self.activation_function(hidden_input)
        #根据第一层的输出数据计算最终输出数据并打印
        final_intputs=numpy.dot(self.who,hidden_outputs)
        final_outputs=self.activation_function(final_intputs)
        return final_outputs
        pass




#输入节点数，隐藏节点数和输出节点数
input_nodes=784
hidden_nodes=100
output_nodes=10
#学习率
learning_rate=0.1

#将参数传递到神经网络中
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)



#打开训练数据

train_data_file=open('mnist_train.csv','r')
train_data_list=train_data_file.readlines()
train_data_file.close()

#训练神经网络
#设置训练的世代数量，将同样的训练数据多次放入神经网络进行计算
epochs=5




for e in range(epochs):
    for records in train_data_list:
        all_values=records.split(',')
        inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
        targets=numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)
        pass
    pass


'''
#打开验证数据
test_data_file=open('','r')
test_data_list=test_data_file.readlines()
test_data_file.close()








#将验证数据输入到训练函数中对模型进行计算
'''








