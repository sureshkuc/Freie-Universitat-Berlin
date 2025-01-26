Calculate the two-sided p-value in R using pnorm and check your results with the command
rank.two.samples

install.packages("coin")
install.packages("rankFD")
library ( coin )
library ( rankFD )

gp1 = c ( 3 , 4  , 4,5 )
gp2 = c ( 3 ,  5 , 6 , 8 , 9 )
two_sided_p_value_calculation_WMW_statistic<-function(gp1,gp2){
    Rik = rank ( c ( gp1 , gp2 ))
    n1 = length ( gp1 )
    n2 = length ( gp2 )
    R1bar = mean( Rik [ 1 : n1 ])
    R2bar = mean( Rik [ n1 + 1 : n2 ])
    N = n1 + n2
    phat = 1 / N * ( R2bar - R1bar ) + 0.5
    sigma0_hat_square = sum (( Rik - ( N + 1 )/ 2 )^ 2 ) / ( N ^ 2 * ( N - 1 ))
    #WMM test statistic
    WN = sqrt (( n1 * n2 ) / ( N * sigma0_hat_square )) * ( phat - 1 / 2 )
    #two sided p-value
    p_value=min( 2 * pnorm ( WN ) , 2 * ( 1 - pnorm ( WN )))
    return(p_value)
}
two_sided_p_value_calculation_WMW_statistic(gp1,gp2)

data = data.frame ( y = c ( gp1 , gp2 ) , grp = factor ( c ( rep ( 1 ,4 ) , rep ( 2 ,5 ))))
rank.two.samples ( y ~ grp , data = data , wilcoxon = "asymptotic" ,
shift.int = FALSE )

Now assume that we had drawn a larger sample, that is n 1 = 40 and n 2 = 50, with the
relative effect and variance estimates p b and σ
b 0 2 remaining unchanged from before. Calculate the
two-sided p-value. Compare and comment on the results.

n1=40
n2=50
N=n1+n2
WN = sqrt (( n1 * n2 ) / ( N * sigma0_hat_square )) * ( phat - 1 / 2 )
#two sided p-value
min ( 2 * pnorm ( WN ) , 2 * ( 1 - pnorm ( WN )))

Excercise-3
Let X 11 , . . . , X 1n 1 ∼ Exp(1) and X 21 , . . . , X 2n 2 ∼ Exp(1) be independent replications from two
independent groups following an exponential distribution with rate parameter 1.
Based on 10,000 simulation runs, approximate the type I error rate in R for the two-sided
asymptotic Wilcoxon-Mann-Whitney test and the two sample t-test assuming equal variances
with nominal significance level 0.05 in case of (a) n 1 = n 2 = 2, (b) n 1 = n 2 = 4, (c) n 1 = n 2 = 7.
Compare and comment on the results.

two_sided_p_value_calculation_t_statistic<-function(gp1,gp2){
    n1 = length ( gp1 )
    n2 = length ( gp2 )
    X1bar = mean( gp1)
    X2bar = mean( gp2)
    N = n1 + n2
    sigma_square = (sum((gp1-X1bar)^2)+sum((gp2-X2bar)^2)) / ( N-2)
    #WMM test statistic
    T_statistic = (X2bar-X1bar)/sqrt ( sigma_square*((1/n1)+(1/n2)))
    p_value=min( 2 * pt ( T_statistic,N-2 ) , 2 * ( 1 - pt ( T_statistic,N-2 )))
    return(p_value)
}

two_sided_p_value_calculation_t_statistic(gp1,gp2)

type_1_error_calculation_with_simulation<-function(n1,n2,number_of_simulation,alpha){
  p_values_WMW <- vector("numeric", number_of_simulation)
  p_values_T <- vector("numeric", number_of_simulation)
  for (i in 1:number_of_simulation){
# Draw exp distributed values
gp1 <- rexp(N, rate = 1)
gp2 <- rexp(N, rate = 1)
p_values_WMW[i]<-two_sided_p_value_calculation_WMW_statistic(gp1,gp2)
#p_values_T[i]<-t.test ( gp1 , gp2 , alternative = "two.sided" , mu = 0 ,paired = FALSE , var.equal = TRUE )[["p.value"]]
p_values_T[i]<-two_sided_p_value_calculation_t_statistic(gp1,gp2)
}
type_1_error_WMW=mean(p_values_WMW<alpha)
type_1_error_T=mean(p_values_T<alpha)
return(list(type_1_error_WMW,type_1_error_T))
}


print(paste0('type 1 error ', list('WMW : ','T Statistic: '), type_1_error_calculation_with_simulation(n1=2,n2=2,number_of_simulation = 10000,alpha=0.05)))

print(paste0('type 1 error ', list('WMW : ','T Statistic: '), type_1_error_calculation_with_simulation(n1=4,n2=4,number_of_simulation = 10000,alpha=0.05)))

print(paste0('type 1 error ', list('WMW : ','T Statistic: '), type_1_error_calculation_with_simulation(n1=7,n2=7,number_of_simulation = 10000,alpha=0.05)))

