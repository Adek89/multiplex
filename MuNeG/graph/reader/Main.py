'''
Created on 2 mar 2014

@author: Adek
'''
from graph.reader.ExcelReader import ExcelReader

FILE_NAME = 'acta_vir'
if __name__ == '__main__':
    er = ExcelReader()
    er.read(FILE_NAME)