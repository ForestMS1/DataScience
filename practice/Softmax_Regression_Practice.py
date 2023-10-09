import torch

x_train = torch.FloatTensor([[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,7,7]])

y_train = torch.FloatTensor([[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0],[0,1,0], [1,0,0],[1,0,0]])

W = torch.randn(4, 3, requires_grad=True)
b = torch.randn(1, 3, requires_grad=True)

optimizer = torch.optim.Adam([W,b], lr=0.1)

for epoch in range(3001):

    h = torch.softmax(torch.mm(x_train, W) + b, dim=1)
    cost = torch.mean(-torch.sum(y_train * torch.log(h), dim=1))

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    with torch.no_grad():
        if epoch % 100 == 0:
            print(f"epcoh: {epoch}, cost: {cost.item()}")


#x가 [1,11,10,9], [1,3,4,3], [1,1,0,1]일때 y값은?

W.requires_grad_(False)
b.requires_grad_(False)

x_test = torch.tensor([[1,11,10,9], [1,3,4,3], [1,1,0,1]],dtype=torch.float)
h_test = torch.softmax(torch.mm(x_test, W) + b, dim=1)
print(h_test)
print(torch.argmax(h_test, dim=1))

