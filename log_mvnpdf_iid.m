% log_mvnpdf_iid: computes mutlivariate normal dist with
%    each dim is iid, so no covariance. 
%   log N(y; mu, diag(d))

function log_p = log_mvnpdf_iid(y, mu, d)

    log_2pi = 1.83787706640934534;
  
    [n, ~] = size(d);
   
    y = y - (mu);

    d_inv = 1 ./ d;
    D_inv_y = d_inv .* y;
  
    K_inv_y = D_inv_y;
  
    log_det_K = sum(log(d));
  
    log_p = -0.5 * (y' * K_inv_y + log_det_K + n * log_2pi);
  end
