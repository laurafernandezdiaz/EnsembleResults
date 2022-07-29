# -*- coding: utf-8 -*-

from .hyperbandcv_coef import HyperbandCV_coef
import numpy as np
import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestRegressor
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
            - rep          : Number of repetitions (default value 1)
            
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
        
        con = sql_connection(self.exp,self.rep)
        
        sql_table_metadata(con)
         
        metadata = (self.X.shape[0],self.X.shape[1],str(str(self.GSparams).count(":")),self.params)#,self.rep)

        sql_insert(con, metadata)
        con.close()

        ids=list(range(0,self.X.shape[0]))
        self.ids_X=np.c_[ids,self.X]
        
    
        if isinstance(self.MLS,RandomForestRegressor):
            resource_param='max_leaf_nodes'
        else:
            resource_param='max_iter'
        self.clf = HyperbandCV_coef(self.MLS, self.GSparams, resource_param=resource_param,cv=self.cv,scoring=self.scoring, pre_dispatch=1,min_iter=int(100000),max_iter=3000000,verbose=1,exp=self.exp,rep=self.rep)#rep=i)
        self.clf.fit(self.ids_X, self.y)
    
        
        self.best_estimator_grid_=self.clf.best_estimator_ #Save the gridsearch coefs in an object variable
        self.fit_times.append(self.clf.fit_times)
        self.score_times.append(self.clf.score_times)
        self.estimators.append(self.clf.estimators)

       
        n_samples = int(self.X.shape[0]) #Number of examples (Xs)
        n_X = int(self.X.shape[1]) #Number of attributes (Xs)
        n_params = int(str(self.GSparams).count(":")) #Number of gridsearch params
        n_params_all = int(self.params) #Number of total values of params
             
        con = sql_connection(self.exp,self.rep)
        self.data_data=sql_fetch_data(con)
        self.data_data=np.asarray(self.data_data)
        con.close()
        
        folds=np.empty((self.data_data.shape[0],1)) #Array with the folds
       
        data = np.empty((self.data_data.shape[0], n_X)) #Array to save the data (Xs)
        y_pred = np.empty((self.data_data.shape[0], 1)) #Array to save the predicted evaluations
        ids = np.empty((self.data_data.shape[0], 1))
        target = np.empty((self.data_data.shape[0],1)) #Array to save the real evaluations
        params = np.empty((self.data_data.shape[0],1),dtype='O') #Array to save the values of params
        reps=np.empty((self.data_data.shape[0], 1)) #Array to save the repetition(id)
        
        conta=0
        #For each file line:
        for i, ir in enumerate(self.data_data):
            
            xs=ir[3].split(';')
            data[i]=np.asarray(xs[1:n_X+1])
            ids[i]=np.asarray(xs[0])
            y_pred[i]=np.asarray(ir[len(ir)-3])
            target[i]=np.asarray(ir[-1])
            params[i]=np.asarray(ir[2])    
            #reps[i]=np.asarray(ir[1]) 
            conta=conta+1
            folds[i]=np.asarray(ir[0]) 
          
        #Concatenate the repetition id, the id of the example, the predicted evaluation, the real evaluation
        self.data_rep=np.c_[ids,folds,data,y_pred,target,params]

        contador=0 
 
        df=pd.DataFrame(self.data_rep)
        df.drop_duplicates(inplace=True)
        self.data_rep=df.to_numpy()
        self.data_rep=self.data_rep[:,:-1]
        self.data_list=[]
        for j in self.data_rep[:,0]:

            self.data_ids_selec=self.data_rep[self.data_rep[:,0]==j] 
            example=np.concatenate((np.array([self.data_ids_selec[0,0]]),np.array([self.data_ids_selec[0,1]]),self.data_ids_selec[0,2:2+n_X],self.data_ids_selec[:,1+n_X+1],np.array([self.data_ids_selec[0,1+n_X+2]])),axis=0) #Concatenate the Xs, the predicted evaluations and the real evaluation                
           
            self.data_list.append(example) #Save each example in a dataset
            contador=contador+1
        
        
        self.data_unique=pd.DataFrame(self.data_list).drop_duplicates().to_numpy()
        #Save the results in variables        
        self.evals_pred=self.data_unique[:,n_X+2:-1]
        self.evals_real=self.data_unique[:,-1]
        self.xs=self.data_unique[:,2:n_X+2]
        self.folds=self.data_unique[:,1]
       

        
