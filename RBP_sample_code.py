# python 2 

from sklearn import datasets
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import time, os, sys, math

bases = ['A', 'C', 'G', 'U']
base_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
bases_len = len(bases)
num_feature = 271 #total number of features


def convert_to_index(str,word_len):
   '''
   convert a sequence 'str' of length 'word_len' into index in 0~4^(word_len)-1
   '''
   output_index = 0
   for i in xrange(word_len):
      output_index = output_index * bases_len + base_dict[str[i]]
   return output_index

def shuffle(trainX,trainY):
   '''
   random shuffle the training data
   '''
   np.random.seed(0)
   train=np.hstack([trainY[:,None],trainX]).astype('float32')
   train=np.random.permutation(train)
   train=train[:120000,:]
   trainX=train[:,1:].astype('float32')
   trainY=train[:,0].astype('float32')
   return trainX,trainY

def leave_out(trainX,trainY):
   '''
   split the data into 70% for training and 30% for testing
   '''
   m=int(0.7*len(trainX))
   testX=trainX[m:,:]
   testY=trainY[m:]
   trainX=trainX[:m,:]
   trainY=trainY[:m]
   return trainX,trainY,testX,testY
   

def extract_features(line):
   '''
   extract features from a sequence of RNA
   
   To do: alternative ways to generate features can be used. 
   '''
   core_seq = line
   for i in 'agctu\n':
      core_seq = core_seq.replace(i, '')
   core_seq = core_seq.replace('T','U')
   core_seq = core_seq.replace('N','')
   final_output=[]
   for word_len in [1,2,3]:
      output_count_list = [0 for i in xrange(bases_len ** word_len)]
      for i in xrange(len(core_seq)-word_len+1):
         output_count_list[convert_to_index(core_seq[i:i+word_len],word_len)] +=1
      final_output.extend(output_count_list)
   return final_output

def load_data(filename,check=False,savecheck='check'):
   '''
   use the extract_features function to extract features for all sequences in the file specified by 'filename'
   '''
   print 'Processing ',filename
   start=time.time()
   total_output=[]
   valid=[]
   for line in open(filename, "r"):
      if line[0] == '>':
         continue
      else:
         if ('n' in line or 'N' in line):
            valid.append(0)
            continue
         else:
            valid.append(1)
            total_output.append(extract_features(line.strip('\n').strip('\r')))
   output_arr=np.array(total_output)
   if (check):
      np.save(savecheck,np.array(valid))
   print output_arr.shape
   end=time.time()
   print 'Finished loading in',end-start,'s\n'
   return output_arr


def training(pathname,dataset):
   '''
   load dataset stored in the directory specified by 'pathname' and then train the model
   '''
   pos_filename=os.path.join(pathname,dataset+'.positives.fa')
   neg_filename=os.path.join(pathname,dataset+'.negatives.fa')
   pos_trainX=load_data(pos_filename)
   pos_trainY=np.ones(len(pos_trainX))
   neg_trainX=load_data(neg_filename)
   neg_trainY=-np.ones(len(neg_trainX))

   trainX=np.vstack([pos_trainX,neg_trainX])
   trainY=np.hstack([pos_trainY,neg_trainY])
   trainX,trainY=shuffle(trainX,trainY)
   #print trainX.shape,trainY.shape,'\n'
   trainX,trainY,testX,testY=leave_out(trainX,trainY)

   print 'Start training...'
   start=time.time()
   '''
   To do: please try other models.
   '''
   model = LogisticRegression(penalty ='l2', C=0.01, n_jobs=16)
   model.fit(trainX,trainY)
   test_pred = model.predict_proba(testX)[:,1]
   roc=roc_auc_score(testY, test_pred)
   
   end=time.time()
   print 'Training ends in',end-start,'s'
   print 'Training roc:',roc,'\n'
   return model

def predicting(filename,model,savepred=None):
   '''
   predict for sequences in 'filename' using the preprocessing transform 'scaler' and the trained model '_model'
   '''
   print "Predicting",filename
   start=time.time()
   if savepred is not None:
      fout=open(savepred,'w')
   try:
      for line in open(filename):
         if line[0]=='>':
            if savepred is not None:
               fout.write(line)
            continue
         elif ('n' in line or 'N' in line):
            if savepred is not None:
               fout.write('Error!\n')
         else:
            line=line.strip('\n').strip('\r')
            testX=np.array(extract_features(line))[None,:]
            pred = model.predict_proba(testX)[:,1]
            if savepred is not None:
               fout.write('%f\n'%float(pred[0]))
   finally:
      if savepred is not None:
         fout.close()



if __name__=='__main__':
    sourcedir = './GraphProt_CLIP_sequences/'
    dataset = './PARCLIP_MOV10_Sievers.train'
    testfile = './GraphProt_CLIP_sequences/PARCLIP_MOV10_Sievers.ls.positives.fa'
    savepred = 'test_output.pred'
		
    '''
    To do: please generate test files for SARS-CoV-2 and make predictions'
    '''
    model = training(sourcedir,dataset)
    predicting(testfile, model, savepred)