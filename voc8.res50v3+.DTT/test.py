# import random

# a=b=c=0

# for i in range(100):
#     index = random.randint(1,3)
#     if index==1:a+=1
#     elif index==2:b+=1
#     elif index==3:c+=1
# print(a)
# print(b)
# print(c)

import torch
import torch.nn as nn
x_input=torch.randn(3,3)
print('x_input:\n',x_input) 
y_target=torch.tensor([0,1,2])


softmax_func=nn.Softmax(dim=1)
soft_output=softmax_func(x_input)
print('soft_output:\n',soft_output)


log_output=torch.log(soft_output)
print('log_output:\n',log_output)


logsoftmax_func=nn.LogSoftmax(dim=1)
logsoftmax_output=logsoftmax_func(x_input)
print('logsoftmax_output:\n',logsoftmax_output)


nllloss_func=nn.NLLLoss()
nlloss_output=nllloss_func(logsoftmax_output,y_target)
print('nlloss_output:\n',nlloss_output)


crossentropyloss=nn.CrossEntropyLoss()
crossentropyloss_output=crossentropyloss(x_input,y_target)
print('crossentropyloss_output:\n',crossentropyloss_output)