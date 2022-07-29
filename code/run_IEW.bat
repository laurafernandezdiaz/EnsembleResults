SET exp=1
REM HPO OPTIONS: GS,RS,Bayesian,PSO,Hyperband
set hpo=GS
mkdir .\IEW%hpo%\%exp%

SET dataname=automobile
SET MLSName=ridge
SET scoringName=neg_mean_squared_error



python.exe IEW%hpo%Regression.py %exp% %dataname% %MLSName% %scoringName% > ./IEW%hpo%/%exp%/%exp%.txt


pause