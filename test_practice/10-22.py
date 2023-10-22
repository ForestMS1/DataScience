import pickle
import matplotlib.pyplot as plt
import numpy as np
data = pickle.load(open("/Users/ms/Desktop/2-2/DataScience/test_practice/mid_animal_data_pub.pkl", "rb"))


"""
for i in range(10) :
    plt.imshow(data['train_images'][i])
    plt.show()
"""



#두 벡터의 내적(Dot Prduct)을 계산
dot_product = np.dot(data['train_vectors'][0],data['test1_vectors'][1]) #두 벡터 내적
#각 벡터의 크기(norm)를 계산
magnitude1 = np.linalg.norm(data['train_vectors'][0])
magnitude2 = np.linalg.norm(data['test1_vectors'][1])
#코사인 유사도 계산
cosine_similarity = dot_product / (magnitude1 * magnitude2)

print(cosine_similarity)

#12
def find_cosr(test1_vector, train_vector,i):
    
    내적 = test1_vector.dot(train_vector)
    test1_vector_size = np.linalg.norm(test1_vector)
    train_vector_size = np.linalg.norm(train_vector)
    cosr=내적/(test1_vector_size*train_vector_size)
    res=(cosr,i) #코사인유사도와 인덱스를 튜플값으로 리턴!
    
    return res
sum = 0 #인덱스 합
resInd=[] #코사인 유사도 젤 높은3개 들어갈곳
for i in range(10):
    res = []

    for j in range(1000):
        res.append(find_cosr(data['test1_vectors'][i],data['train_vectors'][j],j))
    res.sort(reverse=True)
    print(res[0:3])
    for k in range(3):
        resInd.append(res[0:3][k][1])
        sum += res[0:3][k][1]
   

print(resInd)
print(sum)

#13
for i in range(30):
    plt.subplot(10,3,i+1)
    plt.imshow(data['train_images'][resInd[i]])
plt.show()