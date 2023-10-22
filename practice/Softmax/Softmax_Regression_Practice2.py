import torch
import torch.nn.functional as F #cross entropy
import torch.nn as nn

x_train = torch.FloatTensor([[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,7,7]])

y_train = torch.tensor([2,2,2,1,1,1,0,0], dtype=torch.long)

#W = torch.randn(4, 3, requires_grad=True)
#b = torch.randn(1, 3, requires_grad=True)
model = nn.Linear(4,3) #맨날 쓰는 W와 b를 nn.Linear로표현
#optimizer = torch.optim.Adam([W,b], lr=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

for epoch in range(3001):

    #h = torch.softmax(torch.mm(x_train, W) + b, dim=1)
    #cost = torch.mean(-torch.sum(y_train * torch.log(h), dim=1))

    #h = torch.mm(x_train, W) + b
    #cost = F.cross_entropy(h, y_train)
    h = model(x_train)
    cost = F.cross_entropy(h,y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    with torch.no_grad():
        if epoch % 100 == 0:
            print(f"epcoh: {epoch}, cost: {cost.item()}")