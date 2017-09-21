functions {
  /**
    * Return the log probability of a proper conditional autoregressive (CAR) prior 
  * with a sparse representation for the adjacency matrix
  *
    * @param phi Vector containing the parameters with a CAR prior
  * @param tau Precision parameter for the CAR prior (real)
  * @param alpha Dependence (usually spatial) parameter for the CAR prior (real)
  * @param W_sparse Sparse representation of adjacency matrix (int array)
  * @param n Length of phi (int)
  * @param W_n Number of adjacent pairs (int)
  * @param D_sparse Number of neighbors for each location (vector)
  * @param lambda Eigenvalues of D^{-1/2}*W*D^{-1/2} (vector)
  *
    * @return Log probability density of CAR prior up to additive constant
  */
    real sparse_car_lpdf(vector phi, real tau, real alpha, 
                         int[,] W_sparse, vector D_sparse, vector lambda, int n, int W_n) {
      row_vector[n] phit_D; // phi' * D
      row_vector[n] phit_W; // phi' * W
      vector[n] ldet_terms;
      
      phit_D = (phi .* D_sparse)';
      phit_W = rep_row_vector(0, n);
      for (i in 1:W_n) {
        phit_W[W_sparse[i, 1]] = phit_W[W_sparse[i, 1]] + phi[W_sparse[i, 2]];
        phit_W[W_sparse[i, 2]] = phit_W[W_sparse[i, 2]] + phi[W_sparse[i, 1]];
      }
      
      for (i in 1:n) ldet_terms[i] = log1m(alpha * lambda[i]);
      return 0.5 * (n * log(tau)
        + sum(ldet_terms)
        - tau * (phit_D * phi - alpha * (phit_W * phi)));
    }
}

// Calculate incidence per grid cell
data {
  int<lower=0> areas_r; //number of grid cells
  int<lower=0> areas_o; //number of unique observation locations at any level
  int<lower=0> links;   //
  int<lower=0> N;       //number of observations
  real pop[areas_r]; //population in each grid cell
  real water_v[areas_r]; //wash coverage covariate for each grid cell
  real san_v[areas_r]; //wash coverage covariate for each grid cell
  real logpop[areas_r]; //logged population
  real coast_dist[areas_r]; //distance to coastline
  real water_dist[areas_r]; //distance to water(coast,rivers,lakes)
  
  int cases_obs[N]; //observed data at any level
//  real time_obs[N]; //time length of observation in fraction of year
  int grid_cells[links]; //Not necessarily areas_r number of grid cells(only true if no overlap)
  int grid_loc_link[links]; //Vector linking each observation to appropriate location
  int obs_loc_link[N];
 
  //Additional input for sparse matrix approx. of CAR model
  int W_n; //number of adjacent region pairs
  int W_sparse[W_n, 2];   // adjacency pairs
  vector[areas_r] D_sparse;     // diagonal of D (number of neigbors for each site)
  vector[areas_r] eigen_vals;       // eigenvalues of invsqrtD * W * invsqrtD

 
}

parameters{
  real beta0; //non cell specific intercept
  real beta1; //parameter for WASH covariate
  real beta2; //parameter for WASH covariate
  real beta3; //parameter for pop covariate
  real beta4; //parameter for population and water interaction
  real beta5; //parameter for population and sanitation interaction
  real beta6; //parameter for water and sanitation interaction
  real beta7; //parameter for population,water,sanitation interaction 
  real beta8; //parameter for distance to coast
  real beta9; //parameter for distance to any water(coast, lakes, rivers)
  //real epsilon[areas_r]; //spatially independent random effects
  //real psi_nn[areas_r];//Spatial smoothing parameters
  vector[areas_r] phi;//Spatial smoothing parameters
  real<lower=0.0001,upper=1000> tau_nn; //precision of CAR
  real<lower=0.001, upper=1> rho; //spatial autocorr strength param
}

transformed parameters{

  real log_lambda[areas_r]; // individual parameters

  // impose sum-to-zero constraint
  //var_stz = var_nn - mean(var_nn);

  for (j in 1:areas_r) {
    log_lambda[j] = beta0 + phi[j] + beta2 * san_v[j] + beta3 * logpop[j] + beta8 * coast_dist[j] + beta9 * water_dist[j] + beta1 * water_v[j] + beta4 * logpop[j] * water_v[j] + beta5 * logpop[j] * san_v[j] + beta6 * water_v[j] * san_v[j] + beta7 * water_v[j] * san_v[j] * logpop[j];//generates an overdispersed poisson
  }

}

model {
  real exp_cases[N]; //predicted number of cases per cell
  int grid_index; //grid cell index to use when adding grid cells for each observation
  int loc_index;  //location index to use for mapping from observation to grid cells
  int j; //counter
  int k; //counter
  int j_steps;

  beta0 ~ normal(0,10); 
  beta1 ~ normal(-1,10); 
  beta2 ~ normal(-1,10); 
  beta3 ~ normal(0,10);
  beta4 ~ normal(0,10);
  beta5 ~ normal(0,10);
  beta6 ~ normal(0,10);
  beta7 ~ normal(0,10);
  beta8 ~ normal(0,10);
  beta9 ~ normal(0,10);

  //precision matrix for CAR
  //Stern and Cressie (1999) alternative CAR model with varying strength of spatial autocorr
  rho ~ beta(2,1); //comment this line out for a uniform (0,1) prior
  tau_nn ~ gamma(2,2); //gamma(0.5, 0.0005);
  // CAR model using precision version of MVN
  //increment_log_prob(multi_normal_prec_log(var_nn,zeros,Tau));
  phi ~ sparse_car(tau_nn, rho, W_sparse, D_sparse, eigen_vals, areas_r, W_n);
 
  
 
  //model prediction
  //get expected case
k = 1;
j = 1;
j_steps = 0;
while(k <= N){
   exp_cases[k] = 0;
   while(j <= links && grid_loc_link[j] == obs_loc_link[k]){
     grid_index = grid_cells[j];
     exp_cases[k] = exp_cases[k] + exp(log_lambda[grid_index]) *
pop[grid_index];
     j = j + 1;
     j_steps = j_steps + 1;
   }
  k = k + 1;
   if(k<=N && obs_loc_link[k] == obs_loc_link[k-1]) //not done with this location, move back in grid_loc_link
     j = j - j_steps;
   j_steps = 0;
}

  //statistical model of our observations
  cases_obs ~ poisson(exp_cases);  

}