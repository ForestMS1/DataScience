import pandas as pd
import matplotlib.pyplot as plt


seoul = pd.read_csv('/Users/ms/Desktop/2-2/DataScience/test_practice/gts-seoul.csv') #csv파일 읽기
df = pd.DataFrame(seoul) #데이터프레임 생성

cut_col = df[['습도','풍속','현지기압','기온','이슬점 온도']] #컬럼추출

#print(cut_col.dropna()) #nan 버리기
cut_col = cut_col.dropna()
#print(cut_col.min()) 평균,최대,최소
#print(cut_col.corr()) #상관계수?
cut_col = cut_col[cut_col['현지기압']>= 200]
#print(cut_col.corr())

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error #MSE구하려고

x = cut_col[['습도','풍속','현지기압','기온']].values
y = cut_col[['이슬점 온도']].values

lr = LinearRegression()
lr.fit(x,y)
print(lr.coef_,lr.intercept_) #이슬점 온도 계산공식 구한것

y_predict = lr.predict(x) # x를 공식에 넣어서 구한 y예측값

mse = mean_squared_error(y, y_predict) #y실제값과 y예측값의 평균제곱오차 MSE
print(mse)
