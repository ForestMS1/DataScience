import torch

#입력데이터
x_train = torch.FloatTensor([[1],[2],[3],[4],[5],[2.5],[3.5],[0],[3.1],[2.7],[2.8],[2.9]])
y_train = torch.FloatTensor([[1],[1],[1],[0],[0],[0],[0],[1],[0],[1],[1],[1]])

#데이터 시각화---------------------------------
#import matplotlib.pyplot as plt

#plt.scatter(x_train,y_train) #bar, scatter, plot
#plt.show()
#-------------------------------------------
w = torch.randn(1,1)
b = torch.randn(1,1)

lr = 1.0

for epoch in range(3001):
    w.requires_grad_(True) #w.grad = 0 으로 초기화 되어있음
    b.requires_grad_(True) #w와b로 미분할것이다

    #가설함수 sigmoid
    h = torch.sigmoid(x_train @ w + b) #print(h)
    #직접 구현
    #import math
    #h = 1 / (1+math.e ** (-(x_train @ w + b)))

    #코스트 BCE
    #y = 1 --> log(h)
    #y = 0 --> log(1-h)
    cost = torch.mean(-y_train * torch.log(h) - (1-y_train) * torch.log(1-h))  #print(cost)

    cost.backward() #기울기 계산 w.grad -> cost를 w로 미분한결과 / b.grad -> cost를 b로 미분한결과 //w.grad = 0, b.grad = 0 이라는 가정하에 계산함!

    with torch.no_grad():
        w = w - lr * w.grad
        b = b - lr * b.grad

        if epoch % 100 == 0:
            print(epoch, cost.item(), w.item(), b.item())
    

#코스트 BCE를 이렇게 할 수도
# bce = torch.nn.BCELoss()
# cost = bce(h,y_train)


#새로운입력이 들어 왔을때-------------------------
x_test = torch.FloatTensor([[4.5], [1.1]])

test_result = torch.sigmoid(x_test @ w + b)
print(torch.round(test_result)) #반올림
#-------------------------------------------


#그래프그리기
import matplotlib.pyplot as plt

X = torch.linspace(0, 5, 100).unsqueeze(1)
Y = torch.sigmoid(X @ w + b)

plt.scatter(x_train, y_train)
plt.plot(X, Y, c="red")
plt.show()


#sin그래프 그리기
#import math
#X = [x.item() for x in torch.linspace(0, 5, 100)]
#Y = [math.sin(x) for x in X]
#plt.plot(X,Y)
#plt.show()

