import torch

x = torch.tensor([[1,2,3], [4,5,6], [7,8,9]])
#y = torch.FloatTensor([[1,2,3],[4,5,6],[7,8,9]])
y = torch.FloatTensor([[[1,2], [3,4]], [[1,2], [3,4]], [[1,2], [3,4]]])

#print(x.type()) #x타입확인
#print(x.size()) #차원확인
#print(x.shape) #차원확인
#print(x.ndimension())

x = torch.FloatTensor([[[1,2], [3,4]], [[1,2], [3,4]], [[1,2], [3,4]]])
x0 = x.unsqueeze(0) #[3,2,2] --> [1,3,2,2]
x1 = x.unsqueeze(1) #[3,2,2] --> [3,1,2,2]
x2 = x.unsqueeze(2) #[3,2,2] --> [3,2,1,2]
x3 = x.unsqueeze(3) #[3,2,2] --> [3,2,2,1]
print(x0)
print(x1)
print(x2)
print(x3)
print(x3.squeeze().shape) #없는차원을 없애줌
print(x2.squeeze().shape)
print(x1.squeeze().shape)
print(x0.squeeze().shape)
#x.unsqueeeze(0).unsqueeze(1).unsqueeze(0).unsquueze(5) --> 1,1,1,3,2,1,2

print(x.view([2,3,2]))
print(x.view([3,-1]))

#---------------------------------------------------------------------------

x = torch.tensor([[1,2], [3,4], [5,6]], dtype=torch.float) #3by2
w = torch.randn(1,2, dtype = torch.float) #1by2
b = torch.randn(3,1, dtype = torch.float) #3by1

result = torch.mm(x, w.T) + b #행렬곱 -> 3by2 * 2by1 -> 3by1
print(result)

w = torch.tensor(1.0, requires_grad=True) #requires_grad=True ==> w와 관련있는 변수들사이에 어떤 연관관계가있다 저장
a = w*3
l = a**2
l.backward() #w로 미분
print('l을 w로 미분한 값은', w.grad) #w.grad => 기울기
#l = a^2 = (3w)^2 = 9w^2  --> 18w, w=1.0


