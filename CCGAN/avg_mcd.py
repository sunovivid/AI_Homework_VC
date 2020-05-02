import os
import numpy as np

resultFile = open('avg_mcd.txt', 'w')
resultFile.write('testName uniqName epoch MCD\n')
for eachFile in os.listdir('logs_MCD'): #ex) MCD_result_test1_epoch_50
    epoch = eachFile.split('_')[-1].replace('.txt','')
    testName = eachFile
    print('model:{} epoch:{} data calculating...'.format(testName, epoch))
    if 'txt' in eachFile:
        data = open(os.path.join('logs_MCD',eachFile),'r').readlines()
        mcd_list = []
        for eachLine in data: #ex) convType: sptp file: t09_s09 F2 -> M2 MCD: 10.476752754383911
            if eachLine != '\n':
                #print('eachLine: <{}>'.format(eachLine))
                _, _, src, _, trg, _, mcd = eachLine.split(' ')
                #print(' MCD:<{}>'.format(mcd))
                mcd = float(mcd.replace('\n',''))
                mcd_list.append(mcd)
        
        #cal avg
        mcd_list = np.average(mcd_list)
        resultFile.write('{} {} {} \n'.format(testName, epoch, mcd_list))