// Hierarchical logistic model of ABS challenge outcomes (1 = call overturned).
//
//  logit P(overturn) = b0[role] + b1[role] * log(edge_dist)
//                      + sigma_ch[role] * z_ch[challenger] + sigma_u * z_u[umpire]
//
// Each challenger is a unique (player, role) pair, so ch_role gives the role of each.
// role: 1 = batter, 2 = catcher, 3 = pitcher (who initiated the challenge).
// edge_dist: distance of the pitch from the nearest zone edge, in zone-edge units (> 0).
// Random effects are per challenger, with a separate scale per role, so we can ask
// whether catchers really are better challengers, with uncertainty on each player.

data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> y;
  array[N] int<lower=1, upper=3> role;
  vector[N] log_edge_dist;
  int<lower=1> J_ch;                       // unique (player, role) challenger pairs
  array[J_ch] int<lower=1, upper=3> ch_role;
  array[N] int<lower=1, upper=J_ch> ch;
  int<lower=1> J_u;
  array[N] int<lower=1, upper=J_u> ump;
}

parameters {
  vector[3] b0;
  vector[3] b1;
  vector<lower=0>[3] sigma_ch;
  real<lower=0> sigma_u;
  vector[J_ch] z_ch;
  vector[J_u] z_u;
}

model {
  vector[N] eta;
  b0 ~ normal(0, 1.5);
  b1 ~ normal(0, 1);
  sigma_ch ~ normal(0, 0.5);
  sigma_u ~ normal(0, 0.5);
  z_ch ~ std_normal();
  z_u ~ std_normal();

  for (n in 1:N) {
    eta[n] = b0[role[n]] + b1[role[n]] * log_edge_dist[n]
             + sigma_ch[role[n]] * z_ch[ch[n]] + sigma_u * z_u[ump[n]];
  }
  y ~ bernoulli_logit(eta);
}

generated quantities {
  real catcher_minus_batter = b0[2] - b0[1];
  vector[J_ch] challenger_effect;
  for (j in 1:J_ch) challenger_effect[j] = sigma_ch[ch_role[j]] * z_ch[j];
}
