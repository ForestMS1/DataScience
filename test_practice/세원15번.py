import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
data = pickle.load(open("/Users/ms/Desktop/2-2/DataScience/test_practice/mid_animal_data_pub.pkl", "rb"))

x_train = torch.from_numpy(data["train_vectors"])
y_train = torch.from_numpy(data["train_labels"])

model = nn.Linear(256, 3)
optim = torch.optim.Adam(model.parameters(), lr = 0.01)

for epoch in range(10001):
    z = model(x_train) #가설 h
    cost = F.cross_entropy(z,y_train)

    optim.zero_grad()
    cost.backward()
    optim.step()

    with torch.no_grad():
        if epoch % 1000 == 0:
            print(f"epoch:{epoch}, cost:{cost.item()}")

str_labels = ["0", "1", "2"]

def decode_one_hot(one_hot, labels):
    index = np.argmax(one_hot)
    return labels[index]

res = ""
for i in range(30):
    x_test = torch.Tensor(data["test2_vectors"][i])
    test_all = torch.softmax(model(x_test), dim=0)
    res += decode_one_hot(test_all.detach().numpy(), str_labels)
print(res)