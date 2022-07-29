SET exp=1
REM HPO OPTIONS: GridSearch,RandomSearch,Bayesian,PSO,Hyperband
SET hpo=GridSearch
mkdir .\%hpo%\%exp%


SET dataname=automobile
SET MLSName=ridge
SET scoringName=neg_mean_squared_error


pythonw.exe %hpo%Regression.py %exp% %dataname% %MLSName% %scoringName% > ./%hpo%/%exp%/%exp%.txt

pause