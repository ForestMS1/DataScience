import torch

#입력데이터
x_train = torch.FloatTensor([[1],[2],[3],[4],[5],[2.5],[3.5],[0],[3.1],[2.7],[2.8],[2.9]])
y_train = torch.FloatTensor([[1],[1],[1],[0],[0],[0],[0],[1],[0],[1],[1],[1]])

w = torch.randn(1,1, requires_grad = True)
b = torch.randn(1,1, requires_grad = True)

optimizer = torch.optim.SGD([w,b], lr = 1.0) # == torch.optim.Adam([w,b], lr = 1.0) 여러가지 있음

for epoch in range(3001):
    

    #가설함수 sigmoid
    h = torch.sigmoid(x_train @ w + b)

    cost = torch.mean(-y_train * torch.log(h) - (1-y_train) * torch.log(1-h))  #print(cost)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    with torch.no_grad():
        if epoch % 100 == 0:
            print(epoch, cost.item(), w.item(), b.item())

