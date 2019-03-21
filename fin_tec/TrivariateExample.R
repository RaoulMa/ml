#Description: Analyze a Gaussian copula with three random variables.
#Author: Raoul Malm

#First approach to construct a trivariate copula
if(TRUE){
  
library(MASS)
library(psych)
library(rgl)

#fixed random seed
set.seed(100)
        
#trivariate gaussian random values
m <- 3
n <- 2000
sigma <- matrix(c(1, 0.4, 0.2,0.4, 1, -0.8,0.2, -0.8, 1),nrow=3)
print(sigma)
z <- mvrnorm(n,mu=rep(0, m),Sigma=sigma,empirical=T)

#correlation
cor(z,method='spearman')
pairs.panels(z)

#create uniform marginal distributions
u <- pnorm(z)
pairs.panels(u)

#3d plot
plot3d(u[,1],u[,2],u[,3],pch='.',col='blue')

#choose different marginal distributions
x1 <- qgamma(u[,1],shape=2,scale=1)
x2 <- qbeta(u[,2],2,2)
x3 <- qt(u[,3],df=5)
plot3d(x1,x2,x3,pch='.',col='blue')

#correlation
df <- cbind(x1,x2,x3)
cor(df,meth='spearman')
pairs.panels(df)

}

#Second approach for a trivariate copula
if(FALSE){
library(copula)
set.seed(100)
myCop <- normalCopula(param=c(0.4,0.2,-0.8),dim = 3,dispstr = "un")
myMvd <- mvdc(copula=myCop, margins=c("gamma","beta","t"),paramMargins=list(list(shape=2, scale=1),list(shape1=2, shape2=2),list(df=5)))

Z2 <- rMvdc(2000,myMvd)
colnames(Z2) <- c("x1", "x2", "x3")
pairs.panels(Z2)
}










