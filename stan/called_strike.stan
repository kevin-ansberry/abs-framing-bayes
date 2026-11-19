// Hierarchical Bayesian model of called strikes on taken pitches, 2025 vs 2026 (ABS era).
//
//  logit P(called strike) =
//      a0
//    + (beta_d + [season == 2] * beta_dd) * edge_d   linear slope in distance from the zone edge
//    + f_shared(x, z)                 smooth 2D deviation from that slope (HSGP on a grid,
//                                     bilinearly interpolated to each pitch)
//    + [season == 2] * f_delta(x, z)  how the surface changed in the ABS season (spatiotemporal)
//    + tau[month]                     random-walk drift across season-months
//    + u_ump[umpire, season]          umpire effect, separate scale per season
//    + u_cat[catcher, season]         catcher (framing) effect, separate scale per season
//    + u_pit[pitcher] + u_bat[batter]
//
// edge_d = max(|x|, |z|) - 1 is the (Chebyshev) distance outside the nominal zone, in
// zone-edge units. Putting the coarse zone shape in a parametric term lets the GP model only
// small residual structure, which keeps the posterior geometry easy for the sampler.
//
// Key estimands: sigma_c[2] - sigma_c[1] and sigma_u[2] - sigma_u[1] (did ABS shrink
// catcher / umpire effects?), beta_dd (did the edge get sharper?) and the delta surface.

functions {
  // Square root of the 2D squared-exponential spectral density at each basis frequency.
  vector sqrt_spd_se(real alpha, real rho, matrix omega) {
    int M = rows(omega);
    vector[M] out;
    for (j in 1:M) {
      out[j] = sqrt(square(alpha) * 2 * pi() * square(rho)
                    * exp(-0.5 * square(rho) * dot_self(omega[j])));
    }
    return out;
  }
}

data {
  int<lower=1> N;
  int<lower=1> M;
  vector[N] edge_d;                 // distance outside the zone edge (negative inside)
  matrix[M, 2] omega;               // sqrt eigenvalues per basis function
  array[N] int<lower=0, upper=1> y; // 1 = called strike
  array[N] int<lower=1, upper=2> season; // 1 = 2025, 2 = 2026
  int<lower=2> T;
  array[N] int<lower=1, upper=T> tidx;   // season-month index
  int<lower=1> J_u;
  int<lower=1> J_c;
  int<lower=1> J_p;
  int<lower=1> J_b;
  array[N] int<lower=1, upper=J_u> ump;
  array[N] int<lower=1, upper=J_c> cat;
  array[N] int<lower=1, upper=J_p> pit;
  array[N] int<lower=1, upper=J_b> bat;
  int<lower=0> N_ppc;                    // subset used for posterior predictive checks / LOO
  array[N_ppc] int<lower=1, upper=N> ppc_idx;
  int<lower=1> G;                        // grid points for surface plots
  matrix[G, M] Phi_grid;
  vector[G] edge_grid;                   // edge_d on the plotting grid
  // Sparse bilinear interpolation from the grid to each pitch (CSR, 1-indexed).
  int<lower=1> nnz;
  vector[nnz] W_w;
  array[nnz] int<lower=1, upper=G> W_v;
  array[N + 1] int<lower=1> W_u;
}

transformed data {
  vector[N] is2;
  vector[N] is1;
  for (n in 1:N) {
    is2[n] = season[n] == 2 ? 1.0 : 0.0;
    is1[n] = 1.0 - is2[n];
  }
}

parameters {
  real a0;
  real beta_d;                      // slope in edge distance (negative: farther out, fewer strikes)
  real beta_dd;                     // change in that slope in 2026
  real<lower=0> alpha_s;
  real<lower=0> rho_s;
  real<lower=0> alpha_d;
  real<lower=0> rho_d;
  vector[M] zs;
  vector[M] zd;
  vector<lower=0>[2] sigma_u;
  vector<lower=0>[2] sigma_c;
  real<lower=0> sigma_p;
  real<lower=0> sigma_b;
  real<lower=0> sigma_t;
  matrix[J_u, 2] zu;
  matrix[J_c, 2] zc;
  vector[J_p] zp;
  vector[J_b] zb;
  vector[T - 1] zt;
}

model {
  vector[M] sq_s = sqrt_spd_se(alpha_s, rho_s, omega);
  vector[M] sq_d = sqrt_spd_se(alpha_d, rho_d, omega);
  vector[G] g_s = Phi_grid * (sq_s .* zs);
  vector[G] g_d = Phi_grid * (sq_d .* zd);
  vector[N] f_s = csr_matrix_times_vector(N, G, W_w, W_v, W_u, g_s);
  vector[N] f_d = csr_matrix_times_vector(N, G, W_w, W_v, W_u, g_d);
  vector[T] tau = append_row(0, sigma_t * cumulative_sum(zt));
  vector[J_u] u1 = sigma_u[1] * col(zu, 1);
  vector[J_u] u2 = sigma_u[2] * col(zu, 2);
  vector[J_c] c1 = sigma_c[1] * col(zc, 1);
  vector[J_c] c2 = sigma_c[2] * col(zc, 2);
  vector[N] eta = a0 + (beta_d + beta_dd * is2) .* edge_d + f_s + is2 .* f_d + tau[tidx]
                  + is1 .* u1[ump] + is2 .* u2[ump]
                  + is1 .* c1[cat] + is2 .* c2[cat]
                  + sigma_p * zp[pit] + sigma_b * zb[bat];

  // Priors (weakly informative; coordinates are scaled so the zone edge is at +/-1).
  a0 ~ normal(0, 2);
  beta_d ~ normal(-5, 3);
  beta_dd ~ normal(0, 1);
  alpha_s ~ normal(0, 1);           // residual surface: small deviations from the linear slope
  rho_s ~ inv_gamma(5, 2);          // prior mean ~0.5 zone half-widths
  alpha_d ~ normal(0, 0.3);         // season deviation expected to be small
  rho_d ~ inv_gamma(5, 2);
  zs ~ std_normal();
  zd ~ std_normal();
  sigma_u ~ normal(0, 0.5);
  sigma_c ~ normal(0, 0.5);
  sigma_p ~ normal(0, 0.5);
  sigma_b ~ normal(0, 0.5);
  sigma_t ~ normal(0, 0.2);
  to_vector(zu) ~ std_normal();
  to_vector(zc) ~ std_normal();
  zp ~ std_normal();
  zb ~ std_normal();
  zt ~ std_normal();

  y ~ bernoulli_logit(eta);
}

generated quantities {
  real d_sigma_c = sigma_c[2] - sigma_c[1];   // change in catcher-effect scale, 2026 minus 2025
  real d_sigma_u = sigma_u[2] - sigma_u[1];   // change in umpire-effect scale
  vector[J_c] catcher_2025 = sigma_c[1] * col(zc, 1);
  vector[J_c] catcher_2026 = sigma_c[2] * col(zc, 2);
  vector[J_u] umpire_2025 = sigma_u[1] * col(zu, 1);
  vector[J_u] umpire_2026 = sigma_u[2] * col(zu, 2);

  // Called-strike probability surface for an average umpire/catcher, on the plotting grid.
  vector[G] surf_2025;
  vector[G] surf_2026;
  // Posterior predictive checks and pointwise log-likelihood on a random subset.
  vector[N_ppc] log_lik;
  array[N_ppc] int y_rep;
  {
    vector[M] sq_s = sqrt_spd_se(alpha_s, rho_s, omega);
    vector[M] sq_d = sqrt_spd_se(alpha_d, rho_d, omega);
    vector[G] g_s = Phi_grid * (sq_s .* zs);
    vector[G] g_d = Phi_grid * (sq_d .* zd);
    vector[N] f_s_all = csr_matrix_times_vector(N, G, W_w, W_v, W_u, g_s);
    vector[N] f_d_all = csr_matrix_times_vector(N, G, W_w, W_v, W_u, g_d);
    vector[T] tau = append_row(0, sigma_t * cumulative_sum(zt));

    surf_2025 = inv_logit(a0 + beta_d * edge_grid + g_s);
    surf_2026 = inv_logit(a0 + (beta_d + beta_dd) * edge_grid + g_s + g_d);
    for (i in 1:N_ppc) {
      int n = ppc_idx[i];
      int s = season[n];
      real e = a0 + (beta_d + (s == 2 ? beta_dd : 0.0)) * edge_d[n] + f_s_all[n]
               + (s == 2 ? f_d_all[n] : 0.0) + tau[tidx[n]]
               + sigma_u[s] * zu[ump[n], s] + sigma_c[s] * zc[cat[n], s]
               + sigma_p * zp[pit[n]] + sigma_b * zb[bat[n]];
      log_lik[i] = bernoulli_logit_lpmf(y[n] | e);
      y_rep[i] = bernoulli_logit_rng(e);
    }
  }
}
