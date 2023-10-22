from sklearn.linear_model import LinearRegression

x = [[1,2], [3,2], [3,7], [1,1], [1,0]]
y = [[4], [8], [23], [1], [-2]]

lr = LinearRegression() #모델 생성
lr.fit(x, y) # 학습(피팅)

print(lr.coef_, lr.intercept_) #lr.coef_ == w, lr.intercept_ == b


#새로운값
x_test = [[5,10], [2,7], [10,3]]

y_test = lr.predict(x_test)
print(y_test)