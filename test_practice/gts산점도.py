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
print(cut_col)
print(cut_col.corr())





x1 = cut_col[['습도']]
x2 = cut_col[['풍속']]
x3 = cut_col[['현지기압']]
x4 = cut_col[['기온']]
y = cut_col[['이슬점 온도']]

plt.scatter(x1,y)
plt.xlabel("humidity")
plt.ylabel('dew point temperature')
plt.show()

plt.scatter(x2,y)
plt.xlabel("wind speed")
plt.ylabel('dew point temperature')
plt.show()

plt.scatter(x3,y)
plt.xlabel("local pressure")
plt.ylabel('dew point temperature')
plt.show()

plt.scatter(x4,y)
plt.xlabel("temperatures")
plt.ylabel('dew point temperature')
plt.show()
