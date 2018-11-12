import os, os.path
from os import system

print(len(next(os.walk('data'))[1]))

if len(next(os.walk('data'))[1]) >= 3:
	system('python train.py')
else:
    system('python train_one_class_svm.py')