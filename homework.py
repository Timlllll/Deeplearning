import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.show()
tf.__version__
#加载数据集
mnist=tf.keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

print(f"Train image shape:{train_images.shape} Train label shape:{train_labels.shape}");
print(f"Test image shape:{test_images.shape} Test label shape:{test_labels.shape}");

total_num=len(train_images)
valid_split=0.2# 验证集占20%
train_num=int(total_num*(1-valid_split))

train_x=train_images[:train_num]
train_y=train_labels[:train_num]

valid_x=train_images[train_num:]
valid_y=train_labels[train_num:]

test_x=test_images
test_y=test_labels

train_x=train_x.reshape(-1,784)# -1表示不指定，他会在计算过程自动生成
valid_x=valid_x.reshape(-1,784)
test_x=test_x.reshape(-1,784)

train_x=tf.cast(train_x/255.0,tf.float32)
valid_x=tf.cast(valid_x/255.0,tf.float32)
test_x=tf.cast(test_x/255.0,tf.float32)

train_y=tf.one_hot(train_y,depth=10)
valid_y=tf.one_hot(valid_y,depth=10)
test_y=tf.one_hot(test_y,depth=10)

def model(x,w,b):
    pred=tf.matmul(x,w)+b
    return tf.nn.softmax(pred)

W=tf.Variable(tf.random.normal([784,10],mean=0.0,stddev=1.0,dtype=tf.float32))
B=tf.Variable(tf.zeros([10]),dtype=tf.float32)

def loss(x,y,w,b):
    pred=model(x,w,b)
    loss_=tf.keras.losses.categorical_crossentropy(y_true=y,y_pred=pred)
    return tf.reduce_mean(loss_)

training_epochs=20
batch_size=50
lr=0.001

def grad(x,y,w,b):
    with tf.GradientTape() as tape:
        loss_=loss(x,y,w,b)
    return tape.gradient(loss_,[w,b])# 返回梯度向量

optimizer=tf.keras.optimizers.Adam(learning_rate=lr)


def accuracy(x,y,w,b):
    pred=model(x,w,b)
    corr=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    return tf.reduce_mean(tf.cast(corr,tf.float32))


total_step=int(train_num/batch_size)
loss_list_train=[]#train loss
loss_list_valid=[]
acc_list_train=[]#train loss
acc_list_valid=[]

for epoch in range(training_epochs):
    for step in range(total_step):
        xs=train_x[step*batch_size:(step+1)*batch_size,:]
        ys=train_y[step*batch_size:(step+1)*batch_size]
        grads=grad(xs,ys,W,B)#计算梯度
        optimizer.apply_gradients(zip(grads,[W,B]))#优化器调参
    loss_train=loss(train_x,train_y,W,B).numpy()
    loss_valid=loss(valid_x,valid_y,W,B).numpy()
    acc_train=accuracy(train_x,train_y,W,B).numpy()
    acc_vaild=accuracy(valid_x,valid_y,W,B).numpy()
    loss_list_train.append(loss_train)
    loss_list_valid.append(loss_valid)
    acc_list_train.append(acc_train)
    acc_list_valid.append(acc_vaild)
    print('参数W:',W)
    print(f"epoch={epoch+1},train_loss={loss_train},valid_loss={loss_valid},train_accuracy={acc_train},valid_accuracy={acc_vaild}")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(loss_list_train,'blue',label="Train Loss")
plt.plot(loss_list_valid,'red',label='Valid Loss')
plt.legend(loc=1)
plt.show()

acc_test=accuracy(test_x,test_y,W,B).numpy()
print(f'Test acc={acc_test}')