% test_outdata_logpdf.m: test the changes of blue-red model likelihood estimations

% set the global parameters and utility functions
set_parameters;

% load the learned model
variables_to_load = {'bluewards_mu', 'bluewards_sigma', ...
    'redwards_mu', 'redwards_sigma'};
load(sprintf('%s/learned_zqso_only_model_outdata_%s_norm_%d-%d',             ...
    processed_directory(training_release), ...
    training_set_name, normalization_min_lambda, normalization_max_lambda),  ...
    variables_to_load{:});

% mock observation
this_normalized_flux = [1; 2; 3; 4];

% joint log likelihood using pdf
bw_model = makedist('Normal', 'mu', bluewards_mu, 'sigma', bluewards_sigma);
rw_model = makedist('Normal', 'mu', redwards_mu, 'sigma', redwards_sigma);

bw_likelihoods = pdf(bw_model, this_normalized_flux);
bw_log_likelihood = sum(log(bw_likelihoods));

% joint log likelihood using log_mvnpdf_iid
[n, ~] = size(this_normalized_flux);

test_bw_log_likelihood = log_mvnpdf_iid(this_normalized_flux, ...
    bluewards_mu * ones(n, 1), bluewards_sigma^2 * ones(n, 1) );

assert(abs(bw_log_likelihood - test_bw_log_likelihood) < 1e-4 )
