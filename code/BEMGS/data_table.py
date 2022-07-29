# -*- coding: utf-8 -*-

from .gridsearchcv_coef import GridSearchCV_coef
import numpy as np
import sqlite3
from sqlite3 import Error

def sql_connection(exp,rep):

    try:

        con = sqlite3.connect('mydatabase_combineC_'+str(exp)+'_'+str(rep)+'.db')

        return con

    except Error:

        print(Error)

def sql_table_metadata(con):

    cursorObj = con.cursor()

    cursorObj.execute("CREATE TABLE if not exists Metadata (n_examples	INTEGER,n_attributes	INTEGER,n_params	INTEGER,n_all_params	INTEGER)")#,n_reps	INTEGER)")

    con.commit()
    

def sql_insert(con, metadata):

    cursorObj = con.cursor()
    
    cursorObj.execute('INSERT INTO Metadata(n_examples,n_attributes,n_params,n_all_params) VALUES(?, ?, ?, ?)', metadata)
    
    con.commit()
    
def sql_select(con):

    cursorObj = con.cursor()

    cursorObj.execute('SELECT * FROM Metadata')

    con.commit()
    
def sql_fetch_metadata(con):

    cursorObj = con.cursor()

    cursorObj.execute('SELECT * FROM Metadata')

    rows = cursorObj.fetchall()


    return rows

def sql_fetch_data(con):

    cursorObj = con.cursor()

    cursorObj.execute('SELECT * FROM Data')

    rows = cursorObj.fetchall()


    return rows

def sql_drop(con):

    cursorObj = con.cursor()

    cursorObj.execute('DROP table if exists Data')
    con.commit()
    cursorObj.execute('DROP table if exists Metadata')
    con.commit()
    
class data_table:
    def _unique_rows(self,a):
        a = np.ascontiguousarray(a)
        unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
        return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

    def __init__(self,MLS,X,y,GSparams,RG,scoring,total_params,rep,cv_split=2,exp=1):
        """
        Params:
            - MLS    : estimator type
            - GSparams     : params to use in the gridSearch
            - cv           : cross validation object
            - scoring      : score function to use
            
            Dataframe parameters:
            - X            : X data
            - y            : target data
                        
                                      
            Integer parameters:
            - rep          : Number of repetitions
            - cv_split     : Number os cross validation splits
            - exp          : Number of current experiment
            
        """
        #Storing the parameters
        self.MLS=MLS #Estimator
        self.X=X #Data 
        self.y=y #Target        
        
        self.params=total_params #Number of values for all params
        self.GSparams=GSparams #GridSearch params
        self.RG=RG #Cross validatin object
        self.scoring=scoring  #Score function        
        
        self.exp=exp
        self.cv=cv_split
        self.rep=rep
        #Creating the fields
        self.coef_grid_=[] #Aattribute to save the coef of the estimator fitted.       
        self.evals_pred=[] #Attribute to save the predicted evaluations
        self.evals_real=[] #Attribute to save the real evaluations
        self.xs=[] #Attribute to save the examples of Xs
        self.fit_times=[] #Attribute to save the examples of Xs
        self.score_times=[] #Attribute to save the examples of Xs
        self.estimators=[] #Attribute to save the estimators
        con = sql_connection(exp,rep)       
        sql_drop(con)
        con.close()
        
    def fit(self):
        
        "Fit method"
        
        #Write in a file (called eval_grid.csv): number of examples; the number of Xs attributes; number of gridsearch params; number of total values for all params; number of repetitions
        con = sql_connection(self.exp,self.rep)
        
        sql_table_metadata(con)
        metadata = (self.X.shape[0],self.X.shape[1],str(str(self.GSparams).count(":")),self.params)#,self.rep)

        sql_insert(con, metadata)
        con.close()

        ids=list(range(0,self.X.shape[0]))
        self.ids_X=np.c_[ids,self.X]
       
        #Run gridsearch to obtain the evals
        self.clf = GridSearchCV_coef(self.MLS, self.ids_X, self.GSparams, cv=self.cv,scoring=self.scoring,iid=False,verbose=0,cv_split=self.cv,exp=self.exp,rep=self.rep)#rep=i)
        self.clf.fit(self.ids_X, self.y)
    
        
        self.best_estimator_grid_=self.clf.best_estimator_ #Save the gridsearch coefs in an object variable
        self.fit_times.append(self.clf.fit_times)
        self.score_times.append(self.clf.score_times)
        self.estimators.append(self.clf.estimators)
        
       #Read the file that has the gridSearch evaluations

        n_samples = int(self.X.shape[0]) #Number of examples (Xs)
        n_X = int(self.X.shape[1]) #Number of attributes (Xs)
        n_params = int(str(self.GSparams).count(":")) #Number of gridsearch params
        n_params_all = int(self.params) #Number of total values of params
                    
         
        folds=np.tile(range(0,n_samples),n_params_all) #Array with the folds
        #Empty arrays to save the data read.
        data = np.empty((n_params_all*n_samples, n_X)) #Array to save the data (Xs)
        y_pred = np.empty((n_params_all*n_samples, 1)) #Array to save the predicted evaluations
        ids = np.empty((n_params_all*n_samples, 1))
        target = np.empty((n_params_all*n_samples,1)) #Array to save the real evaluations
        params = np.empty((n_params_all*n_samples,1),dtype='O') #Array to save the values of params
        reps=np.empty((n_params_all*n_samples, 1)) #Array to save the repetition(id)
        con = sql_connection(self.exp,self.rep)
        self.data_data=sql_fetch_data(con)
        self.data_data=np.asarray(self.data_data)
        con.close()
        conta=0
        #For each file line:
        for i, ir in enumerate(self.data_data):

            #Obtain the data from file to arrays
            xs=ir[3].split(';')
            data[i]=np.asarray(xs[1:n_X+1])
            ids[i]=np.asarray(xs[0])
            y_pred[i]=np.asarray(ir[len(ir)-3])
            target[i]=np.asarray(ir[-1])
            params[i]=np.asarray(ir[2])    
            conta=conta+1
            folds[i]=np.asarray(ir[0]) 
    

        #Concatenate the repetition id, the id of the example, the predicted evaluation, the real evaluation
        self.data_rep=np.c_[ids,folds,data,y_pred,target]
  
    
        #Create an empty array to save all the data. For each row (Xs, predicted evals, real eval)
        self.data=np.empty((n_params_all*n_samples, n_X+(n_params_all)+3))

        contador=0 #Number of row in Xs (if you use repetitions the xs will be repetitions but the predicted evals will be different. The real eval will be the same)

        for j in self.data_rep[:,0]: #For each example of the selected repetition (for each id)

            self.data_ids_selec=self.data_rep[self.data_rep[:,0]==j] #Select the data from each id. For each repetition there will be as many equal ids as total number of params.
           
            #Concatenate the ids,folds, Xs, the predicted evaluations and the real evaluation  
            example=np.concatenate((np.array([self.data_ids_selec[0,0]]),np.array([self.data_ids_selec[0,1]]),self.data_ids_selec[0,2:2+n_X],self.data_ids_selec[:,1+n_X+1],np.array([self.data_ids_selec[0,1+n_X+2]])),axis=0) #Concatenate the Xs, the predicted evaluations and the real evaluation                
           
            self.data[contador,:]=example #Save each example in a dataset
            contador=contador+1
        
        #Save the results in variables        
        self.data_unique=self._unique_rows(self.data)
        self.evals_pred=self.data_unique[:,n_X+2:-1]
        self.evals_real=self.data_unique[:,-1]
        self.xs=self.data_unique[:,2:n_X+2]
        self.folds=self.data_unique[:,1]

       

        
