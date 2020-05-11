% spectrum_loss: computes the negative log likelihood for centered
% flux y:
%
%     -log p(y | z, σ², M)
%   = -log N(y; 0, MM' + diag(σ²)),
%
% and its derivative wrt M

function [nlog_p, dM] = ...
      spectrum_loss(y, noise_variance, M)

  log_2pi = 1.83787706640934534;

  [n, k] = size(M);

  d = noise_variance;

  d_inv = 1 ./ d;
  D_inv_y = d_inv .* y;
  D_inv_M = bsxfun(@times, d_inv, M);

  % use Woodbury identity, define
  %   B = (I + MᵀD⁻¹M),
  % then
  %   K⁻¹ = D⁻¹ - D⁻¹MB⁻¹MᵀD⁻¹

  B = M' * D_inv_M;
  B(1:(k + 1):end) = B(1:(k + 1):end) + 1;
  L = chol(B);
  % C = B⁻¹MᵀD⁻¹
  C = L \ (L' \ D_inv_M');

  K_inv_y = D_inv_y - D_inv_M * (C * y);

  log_det_K = sum(log(d)) + 2 * sum(log(diag(L)));

  % negative log likelihood:
  %   ½ yᵀ (K + V + A)⁻¹ y + log det (K + V + A) + n log 2π
  nlog_p = 0.5 * (y' * K_inv_y + log_det_K + n * log_2pi);

  % gradient wrt M
  K_inv_M = D_inv_M - D_inv_M * (C * M);
  dM = -(K_inv_y * (K_inv_y' * M) - K_inv_M);

end
