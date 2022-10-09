
## TEMPLATE FOR THE KALMAN FILTER
## This function returns the loglikelihood value which we want to optimize.
Kalman_template<-function(para) #para is a vector containing our set of parameters
{
  #set dt as monthly
  dt<-1/12
  
  # initialize all the parameter values
  kappa11=para[1]
  kappa12=para[2]
  kappa13=para[3]
  kappa14=para[4]
  
  kappa21=para[5]
  kappa22=para[6]
  kappa23=para[7]
  kappa24=para[8]
  
  kappa31=para[9]
  kappa32=para[10]
  kappa33=para[11]
  kappa34=para[12]
  
  kappa41=para[13]
  kappa42=para[14]
  kappa43=para[15]
  kappa44=para[16]
  
  theta1=para[17]
  theta2=para[18]
  theta3=para[19]
  theta4=para[20]
  
  #Force positive
  sigma_NL=abs(para[21])
  sigma_NS=abs(para[22])
  sigma_RL=abs(para[23])
  sigma_RS=abs(para[24])
  
  lambda_N=para[25]
  lambda_R=para[26]
  
  #Note the squared, I.e. our parameter is the standard deviation but we need the variance matrix, so we square it here. Also ensures positivity
  sigma_err_sq=para[27]^2 
  
  # Initialize K^P, \theta^P, H, Sigma
  K=rbind(c(kappa11,kappa12,kappa13,kappa14),
          c(kappa21,kappa22,kappa23,kappa24),
          c(kappa31,kappa32,kappa33,kappa34),
          c(kappa41,kappa42,kappa43,kappa44))
  
  theta=c(theta1,theta2,theta3,theta4)
  
  Sigma=diag(c(sigma_NL,sigma_NS,sigma_RL,sigma_RS))
  
  H=diag(rep(sigma_err_sq,10)) #10 observations per observation date...
  
  # Impose stationarity - I.e. check that eigenvalues of K^P are positive
  # If not let the function return some large value e.g. 999999
  # Save the eigenvalues and vectors of K^P. You will need them later

  #Your implementation
    
  # Calculate the A and B used in the measurement equation. We want to do this outside of the loop!
    
  #Your implementation

  # We calculate the conditional and unconditional covariance matrix

  # Step 1: Calculate the S-overline matrix
    
  #Your implementation
  
  # Step 2: Calculate the V-overline matrix for dt and in the limit

  #Your implementation

  # Step 3: Calculate the final analytical covariance matrices
    
  #Your implementation
    
  # Start the filter at the unconditional mean and variance.

  #Your implementation    

  ## Calculate F_t and C_t
    
  #Your implementation
      
  #Set the initial log likelihood value to zero..
  loglike=0
    
  # Iterate over all the observation dates
  i<-1
  totalNo=??#totalNO is the number of observation dates in your data
    
    while(i<totalNo+1) 
    {
      
      # Perform the prediction step
      
      #Your implementation
      
      # calculate the model-implied yields
      
      #Your implementation
      
      ## Calculate the prefit residuals based on the observed and implied yields
      
      #Your implementation
      
      #Calculate the covariance matrix of the prefit residuals

      #Your implementation
      
      # Calculate the determinant of the covariance matrix of the prefit residuals

      #Your implementation
      
      # Check that the determinant is defined, finite, and positive.
      # If not let the function return some large value e.g. 8888888

      #Your implementation
      
      # Calculate the log determinant

      #Your implementation
      
      # Calculate the inverse of the covariance matrix

      #Your implementation
      
      # Perform the update step

      #Your implementation
      
      #Calculate the ith log likelihood contribution and add the ith log likelihood contribution to the total log likelihood value

      #Your implementation
      
      ## Adding 1 to iteration count
      i=i+1
    }
  #Returning the negative log likelihood (assuming we are minimizing)
  return(-(loglike))
}
