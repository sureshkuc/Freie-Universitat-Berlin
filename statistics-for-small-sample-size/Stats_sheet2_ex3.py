Exercise 3
This is an R exercise based on the built-in data set PlantGrowth. The data result from an
experiment to compare yields (as measured by dried weight of plants) obtained under a control
and two different treatment conditions. Adopting the notation from Exercises 1 and 2, we have
Group 1 (“ctrl”):
Group 2 (“trt1”):
Group 3 (“trt2”):
x 11 , x 12 , . . . , x 1n 1 .
x 21 , x 22 , . . . , x 2n 2 .
x 31 , x 32 , . . . , x 3n 3 .
Compute and report
• the (mid-)ranks of x 11 , . . . , x 1n 1 , x 21 , . . . , x 2n 2 , x 31 , . . . , x 2n 3 among all N = n 1 + n 2 + n 3
observations,
• the mean (mid-)rank in each of the three samples (not the mean internal ranks),
• the relative effects p b 1 , p b 2 , and p b 3 .
In addition, compute and report
• the pseudoranks of x 11 , . . . , x 1n 1 , x 21 , . . . , x 2n 2 , x 31 , . . . , x 2n 3 among all N = n 1 + n 2 + n 3
observations,
• the mean pseudorank in each of the three samples (not the mean internal pseudoranks),
• the relative effects ψ b 1 , ψ b 2 , and ψ b 3 .
Now delete the following 5 observations from the PlantGrowth dataset
26
27
28
29
30
weight group
5.29 trt2
4.92 trt2
6.15 trt2
5.80 trt2
5.26 trt2
and redo this exercise. Compare and comment on the results.

install.packages("rankFD")
data("PlantGrowth")

PlantGrowth

#R-
R_minus<-rank (PlantGrowth$weight , ties.method = "min" )
#R+
R_plus<-rank (PlantGrowth$weight , ties.method = "max" )
#mid-rank for all samples
R_mid<-rank (PlantGrowth$weight , ties.method = "average" ) # default

#mid rank for all samples N=n1+n2+...+n_d
R_mid

summary(PlantGrowth)

library (rankFD)
group = factor ( c ( rep ( 1 ,10 ) , rep ( 2 ,10 ) , rep ( 3 ,10 ) ))
data = data.frame ( response =PlantGrowth$weight , group = group )
# pseudo - ranks :
dataPSR = psr ( response ~ group , data = data , psranks = "Pseudo_Rank" )

dataPSR$Rank=R_mid
dataPSR

# calculation of mean mid-Rank  for individual group
R_i=aggregate(x = dataPSR$Rank,                # Specify data column
          by = list(dataPSR$group),              # Specify group indicator
          FUN = mean) 
          
# average mid rank per group 
R_i

#relative effects p1_hat , p2_hat , and p3_hat
p_i= 1/30*(R_i$x-0.5)

print(paste0('p1_hat',p_i[1]))
print(paste0('p2_hat',p_i[2]))
print(paste0('p3_hat',p_i[3]))

# calculation of mean Pseudo_Rank  for individual group
R_i_Psi=aggregate(x = dataPSR$Pseudo_Rank,                # Specify data column
          by = list(dataPSR$group),              # Specify group indicator
          FUN = mean) 

# average Psi rank per group 
R_i_Psi

Psi_i= 1/25*(R_i_Psi$x-0.5)
##relative effects ψ1_hat , ψ2_hat , and ψ3_hat .
print(paste0('ψ1_hat',Psi_i[1]))
print(paste0('ψ2_hat',Psi_i[2]))
print(paste0('ψ3_hat',Psi_i[3]))

PlantGrowth_after_deletion <- PlantGrowth[c(1:25),]
PlantGrowth_after_deletion

group = factor ( c ( rep ( 1 ,10 ) , rep ( 2 ,10 ) , rep ( 3 ,5 ) ))
data = data.frame ( response =PlantGrowth_after_deletion$weight , group = group )
# pseudo - ranks :
dataPSR = psr ( response ~ group , data = data , psranks = "Pseudo_Rank" )

dataPSR$Rank=rank(PlantGrowth_after_deletion$weight)
dataPSR

# calculation of mean mid-Rank  for individual group
R_i=aggregate(x = dataPSR$Rank,                # Specify data column
          by = list(dataPSR$group),              # Specify group indicator
          FUN = mean) 

# average mid rank per group 
R_i

#relative effects p1_hat , p2_hat , and p3_hat
p_i= 1/25*(R_i$x-0.5)

print(paste0('p1_hat',p_i[1]))
print(paste0('p2_hat',p_i[2]))
print(paste0('p3_hat',p_i[3]))

# calculation of mean Pseudo_Rank  for individual group
R_i_Psi=aggregate(x = dataPSR$Pseudo_Rank,                # Specify data column
          by = list(dataPSR$group),              # Specify group indicator
          FUN = mean) 

# average Psi rank per group 
R_i_Psi

Psi_i= 1/25*(R_i_Psi$x-0.5)
##relative effects ψ1_hat , ψ2_hat , and ψ3_hat .
print(paste0('ψ1_hat',Psi_i[1]))
print(paste0('ψ2_hat',Psi_i[2]))
print(paste0('ψ3_hat',Psi_i[3]))