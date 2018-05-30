#first Python neuralNetwork

import numpy
import
import matplotlib.pyplot




class neuralNetwork:
    #初始化神经网络，输入节点数量，隐藏层节点数量，输出层节点数量喝学习率
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes





        #设置输入层到隐藏层，隐藏层到输出层的权重,权重采用均值为0，标准差为上节点数量平方根的倒数，也就是说节点数量越多，权重越接近0
        self.wih=numpy.random.normal(0.0,pow(self.inodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.onodes,self.hnodes))
        #初始化学习率，学习率介于0-1之间，准确性考虑学习率不应设置过大，面对学习数据较少并且世代数少的神经网络，设置太小也会影响神经网路的准确性
        self.lr=learningrate

        pass



     #训练函数
     def train(self):







         pass



     #验证函数
     def query(self):




         pass