% generate_dla_samples: generates DLA parameter samples from training
% catalog

% load training catalog
catalog = load(sprintf('%s/catalog', processed_directory(training_release)));

% generate quasirandom samples from p(normalized offset, log₁₀(N_HI))
rng('default');
sequence = scramble(haltonset(3), 'rr2');

% the first dimension can be used directly for the uniform prior over
% offsets
offset_samples  = sequence(1:num_dla_samples, 1)';

% ADDING: second dimension for z_qso
offset_samples_qso  = sequence(1:num_dla_samples, 2)';
z_qsos = catalog.z_qsos;
bins = 150;
[z_freq, z_bin] = histcounts(z_qsos, [z_qso_cut : ((max(z_qsos) - z_qso_cut) / bins) : max(z_qsos)]);
for i=length(z_freq):-1:1 z_freq(i) = sum(z_freq(1:i)); end
z_freq = [0 z_freq]; z_freq = z_freq / max(z_freq);
[z_freq, I] = unique(z_freq); z_bin = z_bin(I);
offset_samples_qso = interp1(z_freq, z_bin, offset_samples_qso);

% we must transform the second dimension to have the correct marginal
% distribution for our chosen prior over column density, which is a
% mixture of a uniform distribution on log₁₀ N_HI and a distribution
% we fit to observed data

% uniform component of column density prior
u = makedist('uniform', ...
             'lower', uniform_min_log_nhi, ...
             'upper', uniform_max_log_nhi);

% extract observed log₁₀ N_HI samples from catalog
all_log_nhis = catalog.log_nhis(dla_catalog_name);
ind = cellfun(@(x) (~isempty(x)), all_log_nhis);
log_nhis = cat(1, all_log_nhis{ind});

% make a quadratic fit to the estimated log p(log₁₀ N_HI) over the
% specified range
x = linspace(fit_min_log_nhi, fit_max_log_nhi, 1e3);
kde_pdf = ksdensity(log_nhis, x);
f = polyfit(x, log(kde_pdf), 2);


extrapolate_min_log_nhi = 19.0; % normalization range for the extrapolated region
% convert this to a PDF and normalize
if ~extrapolate_subdla
    unnormalized_pdf = @(nhi) (exp(polyval(f, nhi)));
    Z = integral(unnormalized_pdf, fit_min_log_nhi, 25.0);
else
    unnormalized_pdf = ...
        @(nhi) ( exp(polyval(f,  nhi))       .*      heaviside( nhi - 20.03269 ) ...
        +   exp(polyval(f,  20.03269))  .* (1 - heaviside( nhi - 20.03269 )) );
    Z = integral(unnormalized_pdf, extrapolate_min_log_nhi, 25.0);
end

% create the PDF of the mixture between the uniform distribution and
% the distribution fit to the data
normalized_pdf = @(nhi) ...
          alpha  * (unnormalized_pdf(nhi) / Z) + ...
     (1 - alpha) * (pdf(u, nhi));

 if ~extrapolate_subdla
     cdf = @(nhi) (integral(normalized_pdf, fit_min_log_nhi, nhi));
 else
     cdf = @(nhi) (integral(normalized_pdf, extrapolate_min_log_nhi, nhi));
 end


% use inverse transform sampling to convert the quasirandom samples on
% [0, 1] to appropriate values
log_nhi_samples = zeros(1, num_dla_samples);
for i = 1:num_dla_samples
  log_nhi_samples(i) = ...
      fzero(@(nhi) (cdf(nhi) - sequence(i, 3)), 20.5);
end

% precompute N_HI samples for convenience
nhi_samples = 10.^log_nhi_samples;

variables_to_save = {'uniform_min_log_nhi', 'uniform_max_log_nhi', ...
                     'fit_min_log_nhi', 'fit_max_log_nhi', 'alpha', ...
                     'offset_samples', 'log_nhi_samples', 'nhi_samples', ...
                     'offset_samples_qso'};
save(sprintf('%s/dla_samples', processed_directory(training_release)), ...
     variables_to_save{:}, '-v7.3');
