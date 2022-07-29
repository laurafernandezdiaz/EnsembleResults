

SET exp=1
REM HPO OPTIONS: GS,RS,Bayesian,PSO,Hyperband
set hpo=GS
REM Wrapper OPTIONS: OLS,GEM,Caruana,FS_AICC,FS_AIC,FS_BIC,FS_GMDL,FS_HQIC,PCR_AICC,PCR_AIC,PCR_BIC,PCR_GMDL,PCR_HQIC,PLS_AICC,PLS_AIC,PLS_BIC,PLS_GMDL,PLS_HQIC,BST_AICC,BST_AIC,BST_BIC,BST_GMDL,BST_HQIC,BST_ICM,RBST_AICC,RBST_AIC,RBST_BIC,RBST_GMDL,RBST_HQIC,RBST_ICM
SET wrapper=RBST_ICM
mkdir .\IEW%hpo%\%exp%
mkdir .\Combine%hpo%\%wrapper%\%exp%


SET dataname=automobile
SET MLSName=ridge
SET scoringName=neg_mean_squared_error


pythonw.exe Combine%hpo%Regression.py %exp% %dataname% %MLSName% %scoringName% %wrapper%> ./Combine%hpo%/%wrapper%/%exp%/%exp%.txt


pause