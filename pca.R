# The PCA code is based on
# https://gist.github.com/ahwillia/4c10830640d325e0cab978bc18c6263a
library(tensorflow)
library(ggplot2)

norm<-function(v){sqrt(sum(v^2))}

colNorms<-function(x){
  #compute the L2 norms of columns of a matrix
  apply(x,2,norm)
}

ortho<-function(u,v,ret=c("m","df")){
  #convert factors to orthonormal basis
  #u is LxN, v is LxG
  #returns NxL data frame of factors (similar to PCs) suitable for plotting
  #also returns the GxL loading matrix
  #loading matrix describes how important feature g is in dimension l
  #note we assume columns are samples, rows are features
  #this is the transpose of what prcomp expects
  #eg if we did prcomp(t(Y)) would get t(Y) = "x"%*%t("rotation")
  #the orientation of the returned loadings matches "rotation" from prcomp
  #the orientation of the returned factors matches "x" from prcomp
  #default is to return matrices, can also return data.frames if specified
  ret<-match.arg(ret)
  L<-nrow(u)
  #step 2, convert loadings to orthonormal basis
  svd_v<-svd(v)
  A<-svd_v$u #LxL
  D<-if(length(svd_v$d)>1) diag(svd_v$d) else svd_v$d #LxL
  loadings<-svd_v$v #GxL with orthonormal rows
  factors<-crossprod(u,A%*%D) #NxL
  #step 3, reorder dimensions in decreasing magnitude
  o<-order(colNorms(factors),decreasing=TRUE)
  factors<-factors[,o]
  loadings<-loadings[,o]
  colnames(loadings)<-colnames(factors)<-paste0("dim",1:L)
  if(ret=="df"){
    loadings<-as.data.frame(loadings)
    factors<-as.data.frame(factors)
  }
  mget(c("factors","loadings"))
}

tf_pca<-function(Y,L=2,alpha=1e-4,ortho=TRUE,verbose=TRUE){
  #Y is a data matrix with observations in columns and features in rows
  #Note this is transpose of usual input to prcomp
  #center/scale- should a transformation be applied to the features
  #returns matrices U,V such that Y~=V'U
  #L is dimension of latent space
  #alpha is L2 regularization of U,V
  #ortho=TRUE converts V to orthonormal rows
  #ctl a list of control parameters
  ctl<-list(maxIter=10000,thresh=1e-4,learn_rate=.001)
  #if(center || scale) Y<-t(scale(t(Y),center=center,scale=scale))
  G<-nrow(Y); N<-ncol(Y)
  Y_tf = tf$constant(as.matrix(Y),dtype=tf$float32)
  V<-tf$Variable(tf$random_normal(shape(L,G)))
  U<-tf$Variable(tf$random_normal(shape(L,N)))
  Y_est<-tf$matmul(V,U,transpose_a=TRUE)
  loss<-tf$reduce_sum(tf$squared_difference(Y_tf,Y_est))
  # regularization 
  alpha = tf$constant(alpha)
  regV = alpha*tf$reduce_sum(V^2)
  regU = alpha*tf$reduce_sum(U^2)
  # full objective
  objective = loss + regV + regU
  # optimization setup
  train_step = tf$train$AdamOptimizer(ctl$learn_rate)$minimize(objective)
  # fit the model
  sess<-tf$Session()
  sess$run(tf$global_variables_initializer())
  converged<-FALSE
  obj_trace<-rep(0,ctl$maxIter)
  for(n in seq.int(ctl$maxIter)){
    sess$run(train_step)
    obj_trace[n]<-obj<-sess$run(objective)
    if(verbose && n%%100==0){
      print(paste("iter",n,"objective:",signif(obj,3)))
    }
    if(n>1){
      obj_delta<-abs(obj-obj_old)
      if(obj_delta/obj_old < ctl$thresh){
        converged<-TRUE
        break
      }
    }
    obj_old<-obj
  }
  obj_trace<-obj_trace[1:n]
  V<-sess$run(V)
  U<-sess$run(U)
  sess$close()
  if(ortho){
    res<-ortho(U,V)
    res$obj_trace<-obj_trace
  } else {
    res<-mget(c("U","V","obj_trace"))
  }
  res
}

# N, size of matrix. R, rank of data
N = 100
R = 5

# generate data
W_true <- matrix(rnorm(N*R),nrow=N)
C_true <- matrix(rnorm(N*R),nrow=R)
Y_true <- W_true %*% C_true
res <- tf_pca(Y_true,L=5,ortho=FALSE)
#Y_est<-crossprod(res$V,res$U)
Y_est<-tcrossprod(res$loadings,res$factors)
max(abs(Y_est-Y_true)) #should be a small number
plot(res$obj_trace,type="l") #should show decreasing trend

# Iris Data
#standard PCA
Y<-scale(iris[,1:4])
pca_factors<-as.data.frame(prcomp(Y,rank=2)$x)
colnames(pca_factors)<-paste0("dim",1:2)
pd1<-cbind(pca_factors,sp=iris$Species,alg="pca")

#tensorflow PCA
res_tfpca<-tf_pca(t(Y))
tfpca_factors<-as.data.frame(res_tfpca$factors)
pd2<-cbind(tfpca_factors,sp=iris$Species,alg="tfpca")
pd<-rbind(pd1,pd2)
ggplot(pd,aes(x=dim1,y=dim2,colour=sp))+geom_point()+facet_wrap(~alg)
