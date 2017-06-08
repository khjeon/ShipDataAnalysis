"""
============================
Speed - Power를 a * x ^ b 함수로 curve fitting 하여 결과 비교
============================
"""
from scipy.optimize import curve_fit

def fit_func(x, a, b):
    return a * pow(x, b)

params = curve_fit(fit_func, trainData[0][:,0], trainData[1])
[a, b] = params[0]

return params