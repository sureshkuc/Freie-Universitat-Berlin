X1=c(2.4, 3.0, 3.0, 2.2, 2.2,
2.2, 2.2, 2.8, 2.0, 3.0)

X2=c(
2.8, 2.2, 3.8, 9.4, 8.4,
3.0, 3.2, 4.4, 3.2, 7.4)

xmu1hat<-mean(X1)
xmu2hat<-mean(X2)
deltahat<- xmu1hat-xmu2hat

xmu1hat

xmu2hat

deltahat

xM1hat<-median(X1)
xM2hat<-median(X2)
thetahat<- xM1hat-xM2hat

message('X1:',xM1hat,', X2:',xM2hat,', median:',thetahat)

grid1=expand.grid(X1,X2)

grid1

(grid1[,1]<grid1[,2])

(grid1[,1]<grid1[,2])+1/2*(grid1[,1]==grid1[,2])

prop1 = (grid1[,1]<grid1[,2]) +
1/2*(grid1[,1]==grid1[,2])
phat=mean(prop1)
phat

#Y<X+1/2(X=Y)
prop2 = (grid1[,2]<grid1[,1]) +
1/2*(grid1[,2]==grid1[,1])
qhat=mean(prop2)
qhat


phat+qhat

x3=c(0 , 1)
x4=c(0,0,0,0,0,0,0,0,1,2)
grid2=expand.grid(x3,x4)
prop3 = (grid2[,1]<grid2[,2]) +1/2*(grid2[,1]==grid2[,2]) 
phat1=mean(prop3)
phat1





grid2

X = function ( n ){
rnorm ( 1 , mean = 0 , sd = 1 )
}
x = X(1)

x

x=c(3, 0, 1, 1, 1, 2)
rank (x , ties.method = "min" )
rank (x , ties.method = "max" )
rank (x , ties.method = "average" ) # default

data("PlantGrowth")

placebo = c ( 325 , 375 , 356 , 374 , 412 , 418 , 445 , 379 , 403 , 431 ,
410, 391, 475)
drug = c ( 307 , 268 , 275 , 291 , 314 , 340 , 395 , 279 , 323 , 342 ,
341, 320, 329, 376, 322, 378, 334, 345, 302, 309, 311, 310,
360, 361)

t.test ( drug , placebo , alternative = "two.sided" , mu = 0 ,
paired = FALSE , var.equal = TRUE )

install.packages("BoolNet")

library("BoolNet")

help(stateTransition)

data("cellcycle")

cellcycle

plotNetworkWiring(cellcycle)
initial_state=c(0,0,1,0,0,0,0,0,1,0)
stateTransition(cellcycle,initial_state)
pathatr=getPathToAttractor(cellcycle,initial_state)
pathatr

plotSequence(cellcycle,initial_state)

newstate<-generateState(cellcycle,c("E2F"))

gp1 = c ( 1 , 1 , 3 , 2 , 4 )
gp2 = c ( 2 , 3 , 4 , 5 , 7 , 9 , 9 )
Rik = rank ( c ( gp1 , gp2 ))

Rik

n1 = length ( gp1 )
n2 = length ( gp2 )
R1bar = mean( Rik [ 1 : n1 ])
R2bar = mean( Rik [ n1 + 1 : n2 ])

N = n1 + n2
phat = 1 / N * ( R2bar - R1bar ) + 0.5

phat

sigma0 = sum (( Rik - ( N + 1 )/ 2 )^ 2 ) / ( N ^ 2 * ( N - 1 ))

WN = sqrt (( n1 * n2 ) / ( N * sigma0 )) * ( phat - 1 / 2 )

WN

min ( 2 * pnorm ( WN ) , 2 * ( 1 - pnorm ( WN )))

gp1 = c ( 1 , 1 , 3 , 2 , 4 )
gp2 = c ( 2 , 3 , 4 , 5 , 7 , 9 , 9 )
wilcox.test ( gp1 , gp2 )

install.packages("coin")
install.packages("rankFD")
library ( coin )
library ( rankFD )

library ( coin )
data = data.frame ( y = c ( gp1 , gp2 ) , grp = factor ( c ( rep ( 1 ,5 ) , rep ( 2 ,7 ))))
wilcox_test( y ~ grp , data = data )

?pnorm


library ( rankFD )
rank.two.samples ( y ~ grp , data = data , wilcoxon = "asymptotic" ,
shift.int = FALSE )

gp1 = c ( 3 , 4  , 4,5 )
gp2 = c ( 3 ,  5 , 6 , 8 , 9 )
wilcox.test ( gp1 , gp2 )

data = data.frame ( y = c ( gp1 , gp2 ) , grp = factor ( c ( rep ( 1 ,4 ) , rep ( 2 ,5 ))))
rank.two.samples ( y ~ grp , data = data , wilcoxon = "asymptotic" ,
shift.int = FALSE )

gp1 = c ( 1 , 1.1 , 3 , 2 , 4 )
gp2 = c ( 2.1 , 3.1 , 4.1 ,5, 7, 9, 9.1)

Rik = rank ( c ( gp1 , gp2 ))
R2W = sum ( Rik [ 6 : 12 ])

library ( coin )
data = data.frame( y = c ( gp1 , gp2 ) , grp = factor ( c ( rep ( 1 ,5 ) , rep ( 2 ,7 ))))
wilcox_test ( y ~ grp , data = data , distribution = "exact" )

rank.two.samples ( y ~ grp , data = data , wilcoxon = "exact" ,shift.int = FALSE )

calculateGUI()

gp1 = c ( 1 , 1.1 , 3 , 2 , 4 )
gp2 = c ( 2.1 , 3.1 , 4.1 , 5 , 7 , 9 , 9.1 )
Rik = rank ( c ( gp1 , gp2 ))
R2W = sum ( Rik [ 6 : 12 ])
R2WPermute = c()
for( i in 1 : 100000 ) {
Rikpermute = sample ( Rik ) # Random Permutation
R2WPermute [ i ] = sum ( Rikpermute [ 6 : 12 ]) # rank sum of permuted ran
}
pvalueleft = mean ( R2WPermute <= R2W )
pvalueright = mean ( R2WPermute >= R2W )
pvaluetwo = 2 * min ( pvalueleft , pvalueright )
print ( pvaluetwo )

placebo = c ( 325 , 375 , 356 , 374 , 412 , 418 , 445 , 379 , 403 , 431 ,
410, 391, 475)
drug = c ( 307 , 268 , 275 , 291 , 314 , 340 , 395 , 279 , 323 , 342 ,
341, 320, 329, 376, 322, 378, 334, 345, 302, 309, 311, 310,
360, 361)

data = data.frame ( y = c ( drug , placebo ) ,
grp = factor ( c ( rep ( 1 ,24 ) , rep ( 2 ,13 ))))
rank.two.samples ( y ~ grp , data = data , wilcoxon = "asymptotic" ,shift.int = FALSE )

rank.two.samples ( y ~ grp , data = data , wilcoxon = "exact" ,shift.int = FALSE )

