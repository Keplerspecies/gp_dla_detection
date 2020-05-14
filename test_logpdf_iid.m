% test_logpdf_iid: test the function to calculate iid multivariate normal

load examgrades
x = grades(:,1);

% joint log likelihood using pdf
pd = makedist('Normal','mu',75,'sigma',10);

likelihoods    = pdf(pd, x);
joint_log_likelihood = sum(log(likelihoods));

% joint log likelihood using iid GP likelihood
[n, ~] = size(x);

test_log_likelihoods = log_mvnpdf_iid(x, ...
    ones(n, 1) * 75, ones(n, 1) * 10^2);

assert(abs(test_log_likelihoods - joint_log_likelihood) < 1e-4)
