"""Post-hoc guarantees experiment: the certificate bounds the robust value of any
fixed policy (regardless of origin). Evaluates a fixed policy at full support
(exact V_full^pi) vs projected support (V_proj^pi + eps) and checks
|V_full - V_proj| <= eps, on the new Algorithm/ core."""
