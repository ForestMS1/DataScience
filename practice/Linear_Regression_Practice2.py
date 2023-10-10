import torch

#학습데이터 y = 2x1 + 3x2 -4 의 데이터
x_train = torch.FloatTensor([[1,2], [3,2], [3,7], [1,1], [1,0]])
y_train = torch.FloatTensor([[4], [8], [23], [1], [-2]])

W = torch.randn(2,1) #2by1 #여기서 W.requires_grad_(True)하면 잘 작동x
b = torch.randn(1,1) #1by1

lr = 0.01


for epoch in range(3001):
    W.requires_grad_(True) #반복문안에서 설정
    b.requires_grad_(True) #

    #가설
    h = x_train @ W + b
    #코스트
    cost = ((h-y_train)**2).mean() #MSE

    cost.backward()

    with torch.no_grad() : #밑의 식을 또 미분하면 안돼기때문에 no_grad 설정
        W = W - lr * W.grad 
        b = b - lr * b.grad #b의 기울이의 반대방향으로 lr만큼 곱해서 업데이트

        if epoch % 100 == 0:
            print(epoch, cost.item(), W.squeeze(), b)


#x = [5,10]일때, y의 값은 얼마일까?
x_test = torch.FloatTensor([[5,10]])

y_test = x_test @ W + b # == torch.mm(x_test,W) + b #업데이트해서 찾은 W와 b
print(y_test.item())