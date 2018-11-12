import os, os.path
from os import system

print(len(next(os.walk('data'))[1]))

if len(next(os.walk('data'))[1]) >= 3:
	system('python demo_file.py')
else:
    system('python demo_file_one_class_svm.py')