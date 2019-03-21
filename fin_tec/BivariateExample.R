#Description: Demonstration of Copulas. 
#Author: Raoul Malm

require(mvtnorm)

par(mfrow=c(2,3))

sig = matrix(c(1,0.8,0.8,1),2,2) #correlation matrix
x = rmvnorm(mean=c(0,0),sigma=sig,n=5000) #2 gaussian variables
u = pnorm(x) #uniform
plot(u,pch='.',main = "Gaussian Copula\nrho=0.8")

x1 <- qnorm(u[,1],0,1)
x2 <- qnorm(u[,2],0,1)
plot(x1,x2,pch='.',main="Gaussian Copula\nrho=0.8",xlab="norm(0,1)",ylab="norm(0,1)") 

y1 <- qbeta(u[,1],1,2)
y2 <- qgamma(u[,2],1,2)
plot(y1,y2,pch='.',main="Gaussian Copula\nrho=0.8",xlab="beta(1,2)",ylab="gamma(1,2)") 

sig = matrix(c(1,0.8,0.8,1),2,2) #correlation matrix
x = rmvt(sigma=sig,df=2,n=5000) #2 t-distributed variables
u = pt(x,df=2) #uniform
plot(u,pch='.',main = "Student t-Copula\nrho=0.8,df=2")

x1 <- qnorm(u[,1],0,1)
x2 <- qnorm(u[,2],0,1)
plot(x1,x2,pch='.',main="Student t-Copula\nrho=0.8, df=2",xlab="norm(0,1)",ylab="norm(0,1)") 

y1 <- qbeta(u[,1],1,2)
y2 <- qgamma(u[,2],1,2)
plot(y1,y2,pch='.',main="Student t-Copula\nrho=0.8,df=2",xlab="beta(1,2)",ylab="gamma(1,2)") 


