# This PCA code is a slight modification of
# https://gist.github.com/ahwillia/4c10830640d325e0cab978bc18c6263a
library(tensorflow)

# N, size of matrix. R, rank of data
N = 100
R = 5

# generate data
W_true = matrix(rnorm(N*R),nrow=N)
C_true = matrix(rnorm(N*R),nrow=R)
Y_true = W_true %*% C_true
Y_tf = tf$constant(Y_true,dtype=tf$float32)

W = tf$Variable(tf$random_normal(shape(N,R)))
C = tf$Variable(tf$random_normal(shape(R,N)))
Y_est = tf$matmul(W,C)
loss = tf$reduce_sum((Y_tf-Y_est)^2)

# regularization 
alpha = tf$constant(1e-4)
regW = alpha*tf$reduce_sum(W^2)
regC = alpha*tf$reduce_sum(C^2)

# full objective
objective = loss + regW + regC

# optimization setup
train_step = tf$train$AdamOptimizer(0.001)$minimize(objective)

# fit the model
sess<-tf$Session()
sess$run(tf$global_variables_initializer())
for(n in seq.int(10000)){
  sess$run(train_step)
  if(n %% 100 == 0){
    print(paste("iter",n,"objective:",sess$run(objective)))
  }
}
