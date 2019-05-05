#Description: Model the stock returns with a students t-copula and normal marginals. 
#Author: Raoul Malm

library(copula)
library(VineCopula)

#show more plots at once
par(mfrow=c(3,3))

#set script directory as current work directory 
sourceDir <- getSrcDirectory(function(dummy){dummy})
setwd(sourceDir)

#read in stock return data
cree <- read.csv("cree_r.csv",header=F)$V2
yahoo <- read.csv("yahoo_r.csv",header=F)$V2

#plot distributions
plot(cree,yahoo,pch='.',xlab="Cree returns", ylab = "Yahoo returns",main="Stock returns")
abline(lm(yahoo~cree),col='red',lwd=1)
print(cor(cree,yahoo,method='spearman'))

#analyze data to find out which copula should be used
u <- pobs(as.matrix(cbind(cree,yahoo)))[,1]
v <- pobs(as.matrix(cbind(cree,yahoo)))[,2]
selectedCopula <- BiCopSelect(u,v,familyset=NA)
print(selectedCopula)

#fit t-copula to data and extract rho & df
t.cop <- tCopula(dim=2)
set.seed(500)
m <- pobs(as.matrix(cbind(cree,yahoo)))
fit <- fitCopula(t.cop,m,method='ml')
print(coef(fit))

#show density of t-copula with the extracted rhp & df values
rho <- coef(fit)[1]
df <- coef(fit)[2]
persp(tCopula(dim=2,rho,df=df),dCopula,main="t-Copula density distribution")

#generate random points from such a t-copula
u <- rCopula(3965,tCopula(dim=2,rho,df=df))
plot(u[,1],u[,2],pch='.',col='blue',main="t-Copula density distribution")
cor(u,method='spearman')

#calculate mean and standarddeviations assuming normal marginals
cree_mu <- mean(cree)
cree_sd <- sd(cree)
yahoo_mu <- mean(yahoo)
yahoo_sd <- sd(yahoo)

#compare marginal data with gaussian curves
hist(cree,breaks=80,main='Cree returns',freq=F,density=30,col='blue',ylim=c(0,20),xlim=c(-0.2,0.3))
lines(seq(-0.5,0.5,0.01),dnorm(seq(-0.5,0.5,0.01),cree_mu,cree_sd),col='red',lwd=2)
legend('topright',c('Fitted normal'),col=c('red'),lwd=2)
hist(yahoo,breaks=80,main='Yahoo returns',density=30,col='blue',freq=F,ylim=c(0,20),xlim=c(-0.2,0.2))
lines(seq(-0.5,0.5,0.01),dnorm(seq(-0.5,0.5,0.01),yahoo_mu,yahoo_sd),col='red',lwd=2)
legend('topright',c('Fitted normal'),col=c('red'),lwd=2)

#Create t-copula with fitted rho & df, and use normal marginals with 
#mean and sd as obtained from the empirical data. Then generate random points. 
copula_dist <- mvdc(copula=tCopula(rho,dim=2,df=df), margins=c("norm","norm"),
                    paramMargins=list(list(mean=cree_mu, sd=cree_sd),
                                      list(mean=yahoo_mu, sd=yahoo_sd)))
sim <- rMvdc(3965,copula_dist)

#Compare empirical data with simulated data
plot(cree,yahoo,main='Stock returns')
points(sim[,1],sim[,2],col='red')
legend('bottomright',c('Observed','Simulated'),col=c('black','red'),pch='.')

#choose df=1
set.seed(4258)
copula_dist <- mvdc(copula=tCopula(rho,dim=2,df=1), margins=c("norm","norm"),
                    paramMargins=list(list(mean=cree_mu, sd=cree_sd),
                                      list(mean=yahoo_mu, sd=yahoo_sd)))
sim <- rMvdc(3965,copula_dist)
plot(cree,yahoo,main='Stock returns')
points(sim[,1],sim[,2],col='red')
legend('bottomright',c('Observed','Simulated df=1'),col=c('black','red'),pch='.')
cor(sim[,1],sim[,2],method='spearman')

#choose df=8
copula_dist <- mvdc(copula=tCopula(rho,dim=2,df=8), margins=c("norm","norm"),
                    paramMargins=list(list(mean=cree_mu, sd=cree_sd),
                                      list(mean=yahoo_mu, sd=yahoo_sd)))
sim <- rMvdc(3965,copula_dist)
plot(cree,yahoo,main='Stock returns')
points(sim[,1],sim[,2],col='red')
legend('bottomright',c('Observed','Simulated df=8'),col=c('black','red'),pch='.')
cor(sim[,1],sim[,2],method='spearman')



