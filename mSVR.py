import numpy as np
import sys
from sklearn.svm import SVR#使用支持向量机进行数据驱动
from sklearn.model_selection import GridSearchCV

#基于Iterative Grid-Search的SVR模块
def SVR_IGRID(input, observed, Step=10, Cv=6, threshold=1, convergence=0.5):
  
  assert len(input) == len(observed)
	#初始界限设置
  minedge_C=2**-8
  maxedge_C=2**8
  minedge_gamma=2**-8
  maxedge_gamma=2**8
  
  while 2*Step > threshold:
	#SVR参数设置
    C_range = np.arange(minedge_C, maxedge_C, Step)#优化参数范围设定
    gamma_range = np.arange(minedge_gamma, maxedge_gamma, Step)
    param_grid = dict(gamma=gamma_range, C=C_range)  
	#网格搜索函数设置与SVR拟合
    grid = GridSearchCV(SVR(kernel='rbf'), param_grid=param_grid, scoring='neg_mean_absolute_error', cv=Cv)
    grid.fit(input,observed)
    #参数调整/迭代
    mededge_C=grid.best_params_['C']
    mededge_gamma=grid.best_params_['gamma']
	
    minedge_C=mededge_C-Step-0.5*Step
    minedge_gamma=mededge_gamma-Step-0.5*Step
	
    if minedge_C < 0:
	     minedge_C=2**-8

    if minedge_gamma < 0:
	     minedge_gamma=2**-8

    maxedge_C=mededge_C+Step+convergence*Step
    maxedge_gamma=mededge_gamma+Step+convergence*Step

    Step=convergence*Step
	
  forecast_result=[]
  forecast_result=grid.fit(input,observed).predict(input)
  
  return grid, forecast_result, grid.best_params_['C'], grid.best_params_['gamma']