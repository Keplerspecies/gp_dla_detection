% objective: computes negative log likelihood of entire training
% dataset as a function of the model parameters, x, a vector defined
% as
%
%   x = vec M;
%
% as well as its gradient:
%
%   f(x) = -∑ᵢ log(yᵢ | z, σ², M)
%   g(x) = ∂f/∂x

function [f, g] = objective(x, centered_rest_fluxes, rest_noise_variances)

  [num_quasars, num_pixels] = size(centered_rest_fluxes);

  k = numel(x) / num_pixels;

  M = reshape(x, [num_pixels, k]);

  f          = 0;
  dM         = zeros(size(M));

  for i = 1:num_quasars
    ind = (~isnan(centered_rest_fluxes(i, :)));

    [this_f, this_dM] = spectrum_loss(centered_rest_fluxes(i, ind)', rest_noise_variances(i, ind)', M(ind, :));

    f               = f               + this_f;
    dM(ind, :)      = dM(ind, :)      + this_dM;
  end

  g = dM(:,1);

end
