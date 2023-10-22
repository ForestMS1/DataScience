import torch

x_train = torch.FloatTensor([[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,7,7]])

y_train = torch.FloatTensor([[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0],[0,1,0], [1,0,0],[1,0,0]])

W = torch.randn(4, 3, requires_grad=True) #여기서 W는 matrix임 row는 x의차원과 같아야하고 col은 클래스의개수
b = torch.randn(1, 3, requires_grad=True) #클래스가 k개면 torch.randn(1,k)

optimizer = torch.optim.Adam([W,b], lr=0.1)

for epoch in range(3001):

    h = torch.softmax(torch.mm(x_train, W) + b, dim=1) #dim=1 1차원에 소프트맥스
    cost = torch.mean(-torch.sum(y_train * torch.log(h), dim=1))

    optimizer.zero_grad() #기울기 초기화
    cost.backward() #기울기 계산
    optimizer.step() #업데이트

    with torch.no_grad():
        if epoch % 100 == 0:
            print(f"epcoh: {epoch}, cost: {cost.item()}")


#x가 [1,11,10,9], [1,3,4,3], [1,1,0,1]일때 y값은?-------------------------

W.requires_grad_(False)
b.requires_grad_(False)

x_test = torch.tensor([[1,11,10,9], [1,3,4,3], [1,1,0,1]],dtype=torch.float)
h_test = torch.softmax(torch.mm(x_test, W) + b, dim=1)  #W와 b는 위에서 구했으니까 그대로 사용
print(h_test)
print(torch.argmax(h_test, dim=1)) #벡터안의 값중에 제일 큰 원소의 위치를 리턴

