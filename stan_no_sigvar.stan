data {
  int<lower=1> N;  // no. observations   
  int<lower=0> p;  // number of x + 1 for intercept (this needs to be a matrix of 1s.)
  vector[N] y;     //  log POC
  matrix[N,p] x;   // matrix of x values
}
parameters{
  vector[p] beta;
}

model{
  // Priors
  //?????

  // Likelihood
  for(i in 1:N){
  y[i] ~ normal(x[i,]*beta,0.0001);
  }
}
generated quantities{
  array[N] real y_sim;

  for(i in 1:N){
  y_sim[i] = normal_rng(x[i,]*beta,0.0001);
  }
}