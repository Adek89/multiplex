'''
Created on 8 mar 2014

@author: Adek
'''

class EvaluationTools:
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
        '''
        Method calculate accuracy
        @param originalLabels: array of original node labels
        @param estimatedLabels: array of estimated node labels
        '''
    def calculateAccuracy(self, originalLabels, estimatedLabels):
        goodRecognitions = 0
        numberOfLabels = originalLabels.__len__()
        for i in range(1, numberOfLabels):
            if (originalLabels[i] == estimatedLabels[i]):
                goodRecognitions = goodRecognitions + 1
        return float(goodRecognitions) / float(numberOfLabels)
    '''
    Method calculate confusion matrix
    @param originalLabels: array of original node labels
    @param estimatedLabels: array of estimated node labels
    @param numberOfClasses: number of classes that nodes can be labeled
    
    @return: confusion matrix
    
    '''      
    
    def calculateConfusionMatrix(self, originalLabels, estimatedLabels, numberOfClasses):
        confusionMatrix = []
        for i in range(0, numberOfClasses):
            currRow = []
            for j in range(0, numberOfClasses):
                currRow.append(0)
            confusionMatrix.append(currRow)  
        
        for i in range(0, originalLabels.__len__()):
            estimatedElement = estimatedLabels.__getitem__(i)
            originalElement = originalLabels.__getitem__(i)
            row = confusionMatrix.__getitem__(estimatedElement)   
            value = row.__getitem__(originalElement)
            row[originalElement] = value + 1
        return confusionMatrix
    
    
    '''
    Method calculate confusion table
    @param originalLabels: array of original node labels
    @param estimatedLabels: array of estimated node labels
    @param numberOfClasses: number of classes that nodes can be labeled
    @param classNr: Number of actual class
    
    @return: confusion table
    
    '''  
    def calculateConfusionTable(self, originalLabels, estimatedLabels, numberOfClasses, classNr):
        confusionMatrix = self.calculateConfusionMatrix(originalLabels, estimatedLabels, numberOfClasses)
        confusionTable = [[0, 0], [0, 0]]
        truePositive = confusionMatrix[classNr][classNr]
        
        falseNegative = 0
        for i in range(0, numberOfClasses):
            falseNegative = falseNegative + confusionMatrix.__getitem__(i).__getitem__(classNr)
        falseNegative = falseNegative - truePositive
        
        falsePositive = 0
        
        for j in range(0, numberOfClasses):
            falsePositive = falsePositive + confusionMatrix.__getitem__(i).__getitem__(classNr)
        falsePositive = falsePositive - truePositive
        
        total = 0
        for i in range(0, numberOfClasses):
            for j in range(0, numberOfClasses):
                total = total + confusionMatrix.__getitem__(i).__getitem__(j)
        trueNegatives = total - truePositive - falseNegative - falsePositive
        
        confusionTable[0][0] = truePositive
        confusionTable[1][0] = falseNegative
        confusionTable[0][1] = falsePositive
        confusionTable[1][1] = trueNegatives
        
        return confusionTable
   
    def calculateFMeasure(self, originalLabels, estimatedLabels, numberOfClasses, classNr):
        tp = 0
        fn = 0
        fp = 0
        tn = 0
           
        confusionTable = self.calculateConfusionTable(originalLabels, estimatedLabels, numberOfClasses, classNr)
           
        tp = tp + confusionTable.__getitem__(0).__getitem__(0)
        fn = fn + confusionTable.__getitem__(1).__getitem__(0)
        fp = fp + confusionTable.__getitem__(0).__getitem__(1)
        tn = tn + confusionTable.__getitem__(1).__getitem__(1)
           
       
        precision = self.calculatePrecision(confusionTable, originalLabels, estimatedLabels, numberOfClasses, classNr)
        recall = self.calculateRecall(confusionTable, originalLabels, estimatedLabels, numberOfClasses, classNr)
        f = (2.0 * precision * recall)/(1.0*precision+recall)
           
        return f
    
    def calculatePrecision(self, confusionTable, originalLabels, estimatedLabels, numberOfClasses, classNr):
        tp = 0
        fp = 0  
         
        tp = tp + confusionTable.__getitem__(0).__getitem__(0)  
        fp = fp + confusionTable.__getitem__(0).__getitem__(1)
        precision = float(tp)/float(tp+fp)
        return precision
    
    def calculateRecall(self, confusionTable, originalLabels, estimatedLabels, numberOfClasses, classNr):
        tp = 0
        fn = 0
        
        tp = tp + confusionTable.__getitem__(0).__getitem__(0)
        fn = fn + confusionTable.__getitem__(1).__getitem__(0)
        
        recall = float(tp)/float(tp+fn)
        return recall
    
    def calculateFMacro(self, originalLabels, estimatedLabels, numberOfClasses):
        tp = []
        fp = []
        tn = []
        fn = []
        fMeasure = []
        for i in range(0, numberOfClasses):
            tp.append(0.0)
            fp.append(0.0)
            tn.append(0.0)
            fn.append(0.0)
            fMeasure.append(0.0)
        fMeasureResult = 0.0
        
        for i in range(0, numberOfClasses):
            for j in range(0, originalLabels.__len__()):
                if (originalLabels.__getitem__(j) == i and estimatedLabels.__getitem__(j) == i):
                    tp[i] = tp.__getitem__(i) + 1
                elif (originalLabels.__getitem__(j) != i and estimatedLabels.__getitem__(j) == i):
                    fp[i] = fp.__getitem__(i) + 1
                elif (originalLabels.__getitem__(j) != i and estimatedLabels.__getitem__(j) != i):
                    tn[i] = tn.__getitem__(i) + 1
                elif (originalLabels.__getitem__(j) == i and estimatedLabels.__getitem__(j) != i):
                    fn[i] = fn.__getitem__(i) + 1
        for i in range(0, numberOfClasses):
            fMeasure[i] = (2*tp[i])/(2*tp[i]+1*fp[i]+fn[i])
            fMeasureResult = fMeasureResult + fMeasure[i]
        fMeasureResult = fMeasureResult / float(numberOfClasses)
        
        return fMeasureResult   
    
    def calculateFMacro2(self, originalLabels, estimatedLabels, numberOfClasses):
        fMacro = 0.0
        for i in range(0, numberOfClasses):
            fMacro = fMacro + self.calculateFMeasure(originalLabels, estimatedLabels, numberOfClasses, i)
        fMacro = fMacro / float(numberOfClasses)
        return fMacro
    
    def calculateFMicro(self, originalLabels, estimatedLabels, numberOfClasses):
        tp = []
        fp = []
        tn = []
        fn = []
        tpSum = 0.0
        fnSum = 0.0
        fpSum = 0.0
        tnSum = 0.0
        fMeasure = 0.0
        for i in range(0, numberOfClasses):
            tp.append(0.0)
            fp.append(0.0)
            tn.append(0.0)
            fn.append(0.0)
        
        for i in range(0, numberOfClasses):
            for j in range(estimatedLabels.__len__()):
                if (originalLabels.__getitem__(j) == i and estimatedLabels.__getitem__(j) == i):
                    tp[i] = tp.__getitem__(i) + 1.0
                elif (originalLabels.__getitem__(j) != i and estimatedLabels.__getitem__(j) == i):
                    fp[i] = fp.__getitem__(i) + 1.0
                elif (originalLabels.__getitem__(j) != i and estimatedLabels.__getitem__(j) != i):
                    tn[i] = tn.__getitem__(i) + 1.0
                elif (originalLabels.__getitem__(j) == i and estimatedLabels.__getitem__(j) != i):
                    fn[i] = fn.__getitem__(i) + 1.0
                    
        for i in range(0, numberOfClasses):
            tpSum = tpSum + tp[i]
            fpSum = fpSum + fp[i]
            tnSum = tnSum + tn[i]
            fnSum = fnSum + fn[i]
        fMeasure = (2 * tpSum) / (2*tpSum+1*fpSum+fnSum)
        
        return fMeasure
    
    def calculateFMicro2(self, originalLabels, estimatedLabels, numberOfClasses):
            fMicro = 0.0
            tp = 0.0
            fn = 0.0
            fp = 0.0
            tn = 0.0
            confusionTable = []
            
            for i in range(0, numberOfClasses):
                confusionTable = self.calculateConfusionTable(originalLabels, estimatedLabels, numberOfClasses, i)
                tp = tp + confusionTable.__getitem__(0).__getitem__(0)
                fn = fn + confusionTable.__getitem__(1).__getitem__(0)
                fp = fp + confusionTable.__getitem__(0).__getitem__(1)
                tn = tn + confusionTable.__getitem__(1).__getitem__(1)
            
            precision = tp / float(tp+fp)
            recall = tp / float(tp+fn)
            
            fMicro = (2.0 * precision * recall)/(1.0 *precision+recall)
            
            return fMicro
    
    