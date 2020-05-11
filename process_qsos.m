% process_qsos: run DLA detection algorithm on specified objects
% 
% Apr 8, 2020: add all Lyman series to the effective optical depth
%   effective_optical_depth := ∑ τ fi1 λi1 / ( f21 λ21 ) * ( 1 + z_i1 )^β
%  where 
%   1 + z_i1 =  λobs / λ_i1 = λ_lya / λ_i1 *  (1 + z_a)
% Dec 25, 2019: add Lyman series to the noise variance training
%   s(z)     = 1 - exp(-effective_optical_depth) + c_0 
% the mean values of Kim's effective optical depth
prev_tau_0 = 0.0023;
prev_beta  = 3.65;

% load redshifts/DLA flags from training release
prior_catalog = ...
    load(sprintf('%s/catalog', processed_directory(training_release)));

if (ischar(prior_ind))
    prior_ind = eval(prior_ind);
end

prior.z_qsos  = prior_catalog.z_qsos(prior_ind);
prior.dla_ind = prior_catalog.dla_inds(dla_catalog_name);
prior.dla_ind = prior.dla_ind(prior_ind);

% filter out DLAs from prior catalog corresponding to region of spectrum below
% Ly∞ QSO rest
prior.z_dlas = prior_catalog.z_dlas(dla_catalog_name);
prior.z_dlas = prior.z_dlas(prior_ind);

for i = find(prior.dla_ind)'
    if (observed_wavelengths(lya_wavelength, prior.z_dlas{i}) < ...
            observed_wavelengths(lyman_limit,    prior.z_qsos(i)))
        prior.dla_ind(i) = false;
    end
end

prior = rmfield(prior, 'z_dlas');

% load QSO model from training release
variables_to_load = {'rest_wavelengths', 'mu', 'M', 'log_omega', ...
    'log_c_0', 'log_tau_0', 'log_beta', ...
    'bluewards_mu', 'bluewards_sigma', ...
    'redwards_mu', 'redwards_sigma'};
load(sprintf('%s/learned_qso_model_%s',             ...
    processed_directory(training_release), ...
    training_set_name),                    ...
    variables_to_load{:});

% load DLA samples from training release
variables_to_load = {'offset_samples', 'offset_samples_qso', 'log_nhi_samples', 'nhi_samples'};
load(sprintf('%s/dla_samples', processed_directory(training_release)), ...
    variables_to_load{:});

% load redshifts from catalog to process
catalog = load(sprintf('%s/catalog', processed_directory(release)));

% load preprocessed QSOs
variables_to_load = {'all_wavelengths', 'all_flux', 'all_noise_variance', ...
    'all_pixel_mask'};
load(sprintf('%s/preloaded_qsos', processed_directory(release)), ...
    variables_to_load{:});

% enable processing specific QSOs via setting to_test_ind
if (ischar(test_ind))
    test_ind = eval(test_ind);
end

all_wavelengths    =    all_wavelengths(test_ind);
all_flux           =           all_flux(test_ind);
all_noise_variance = all_noise_variance(test_ind);
all_pixel_mask     =     all_pixel_mask(test_ind);
all_thing_ids      =   catalog.thing_ids(test_ind);

z_qsos = catalog.z_qsos(test_ind);
dla_inds = catalog.dla_inds('dr12q_visual');
dla_inds = dla_inds(test_ind);

num_quasars = numel(z_qsos);
if exist('qso_ind', 'var') == 0
    qso_ind = 1:1:floor(num_quasars/100);
end
num_quasars = numel(qso_ind);

% preprocess model interpolants
mu_interpolator = ...
    griddedInterpolant(rest_wavelengths,        mu,        'linear');
M_interpolator = ...
    griddedInterpolant({rest_wavelengths, 1:k}, M,         'linear');
log_omega_interpolator = ...
    griddedInterpolant(rest_wavelengths,        log_omega, 'linear');

%preprocess Gaussian i.i.d. model for off-restframe observations
%bw_model = makedist('Normal', 'mu', bluewards_mu, 'sigma', bluewards_sigma);
%rw_model = makedist('Normal', 'mu', redwards_mu, 'sigma', redwards_sigma);

% initialize results
% prevent parfor error, should use nan(num_quasars, num_dla_samples); or not save these variables;
min_z_dlas                    = nan(num_quasars, num_dla_samples);
max_z_dlas                    = nan(num_quasars, num_dla_samples);
% sample_log_priors_no_dla      = nan(num_quasars, num_dla_samples); % comment out these to save memory
% sample_log_priors_dla         = nan(num_quasars, num_dla_samples);
% sample_log_likelihoods_no_dla = nan(num_quasars, num_dla_samples);
% sample_log_likelihoods_dla    = nan(num_quasars, num_dla_samples);
log_posteriors_no_dla         = nan(num_quasars, 1);
log_posteriors_dla_sub        = nan(num_quasars, 1);
log_posteriors_dla_sup        = nan(num_quasars, 1);
log_posteriors_dla            = nan(num_quasars, 1);
z_true                        = nan(num_quasars, 1);
dla_true                      = nan(num_quasars, 1);
z_map                         = nan(num_quasars, 1);
z_dla_map                     = nan(num_quasars, 1);
n_hi_map                      = nan(num_quasars, 1);
signal_to_noise               = nan(num_quasars, 1);

c_0   = exp(log_c_0);
tau_0 = exp(log_tau_0);
beta  = exp(log_beta);

z_list                   = 1:length(offset_samples_qso);
%Debug output
%all_mus = cell(size(z_list));

fluxes                   = cell(length(z_list), 1);
rest_wavelengths         = cell(length(z_list), 1);
this_p_dlas              = zeros(length(z_list), 1);

quasar_ind = 1;
try
    load(['./checkpointing/curDLA_', optTag, '.mat']); %checkmarking code
catch ME
    0;
end
q_ind_start = quasar_ind;

% catch the exceptions
all_exceptions = false(num_quasars, 1);

for quasar_ind = q_ind_start:num_quasars %quasar list
    tic;
    quasar_num = qso_ind(quasar_ind);
    
    z_true(quasar_ind)   = z_qsos(quasar_num);
    dla_true(quasar_ind) = dla_inds(quasar_num);
    fprintf('processing quasar %i/%i (z_true = %0.4f) ...', ...
        quasar_ind, num_quasars, z_true(quasar_ind));

    %computing signal-to-noise ratio
    this_wavelengths    =    all_wavelengths{quasar_num};
    this_flux           =           all_flux{quasar_num};
    this_noise_variance = all_noise_variance{quasar_num};
    this_pixel_mask     =     all_pixel_mask{quasar_num};

    this_rest_wavelengths = emitted_wavelengths(this_wavelengths, 4.4088); %roughly highest redshift possible (S2N for everything that may be in restframe)

    ind  = this_rest_wavelengths <= max_lambda;

    this_rest_wavelengths = this_rest_wavelengths(ind);
    this_flux             =             this_flux(ind);
    this_noise_variance   =   this_noise_variance(ind);

    this_noise_variance(isinf(this_noise_variance)) = nanmean(this_noise_variance); %kludge to fix bad data
    
    this_pixel_signal_to_noise  = sqrt(this_noise_variance) ./ abs(this_flux);

    % this is before pixel masking; nanmean to avoid possible NaN values
    signal_to_noise(quasar_ind) = nanmean(this_pixel_signal_to_noise);

    % this is saved for the MAP esitmate of z_QSO
    used_z_dla                         = nan(num_dla_samples, 1);

    % initialise some dummy arrays to reduce memory consumption 
    this_sample_log_priors_no_dla      = nan(1, num_dla_samples);
    this_sample_log_priors_dla         = nan(1, num_dla_samples);
    this_sample_log_likelihoods_no_dla = nan(1, num_dla_samples);
    this_sample_log_likelihoods_dla    = nan(1, num_dla_samples);

    % move these outside the parfor to avoid constantly querying these large arrays
    this_out_wavelengths    =    all_wavelengths{quasar_num};
    this_out_flux           =           all_flux{quasar_num};
    this_out_noise_variance = all_noise_variance{quasar_num};
    this_out_pixel_mask     =     all_pixel_mask{quasar_num};

    % Test: see if this spec is empty; this error handling line be outside parfor
    % would avoid running lots of empty spec in parallel workers
    if all(size(this_out_wavelengths) == [0 0])
        all_exceptions(quasar_ind, 1) = 1;
        continue;
    end
    
    for i = 1:num_dla_samples       %variant redshift in quasars 
     
        z_qso = offset_samples_qso(i);
        if mod(i, 250) == 0
            disp(i);
        %end
        % keep a copy inside the parfor since we are modifying them
        this_wavelengths    = this_out_wavelengths;
        this_flux           = this_out_flux;
        this_noise_variance = this_out_noise_variance;
        this_pixel_mask     = this_out_pixel_mask;
        
        %Cut off observations
        max_pos_lambda = observed_wavelengths(max_lambda, z_qso);
        min_pos_lambda = observed_wavelengths(min_lambda, z_qso);
        max_observed_lambda = min(max_pos_lambda, max(this_wavelengths));
        min_observed_lambda = max(min_pos_lambda, min(this_wavelengths));
        lambda_observed = (max_observed_lambda - min_observed_lambda);
        
        %Find probability for out-of-range model
        %bw_likelihoods = pdf(bw_model, this_flux(this_wavelengths < min_observed_lambda));
        %bw_log_likelihood = sum(log(bw_likelihoods));
        %rw_likelihoods = pdf(rw_model, this_flux(this_wavelengths > max_observed_lambda));
        %rw_log_likelihood = sum(log(rw_likelihoods));
        
        ind = (this_wavelengths > min_observed_lambda) & (this_wavelengths < max_observed_lambda);
        this_flux           = this_flux(ind);
        this_noise_variance = this_noise_variance(ind);
        this_wavelengths    = this_wavelengths(ind);
        this_pixel_mask     = this_pixel_mask(ind);

        % convert to QSO rest frame
        this_rest_wavelengths = emitted_wavelengths(this_wavelengths, z_qso);

        %normalizing here
        ind = (this_rest_wavelengths >= normalization_min_lambda) & ...
            (this_rest_wavelengths <= normalization_max_lambda);

        this_median         = nanmedian(this_flux(ind));
        this_flux           = this_flux / this_median;
        this_noise_variance = this_noise_variance / this_median .^ 2;

        ind = (this_rest_wavelengths >= min_lambda) & ...
            (this_rest_wavelengths <= max_lambda);

        % keep complete copy of equally spaced wavelengths for absorption
        % computation
        this_unmasked_wavelengths = this_wavelengths(ind);

        ind = ind & (~this_pixel_mask);

        this_wavelengths      =      this_wavelengths(ind);
        this_rest_wavelengths = this_rest_wavelengths(ind);
        this_flux             =             this_flux(ind);
        this_noise_variance   =   this_noise_variance(ind);

        this_noise_variance(isinf(this_noise_variance)) = mean(this_noise_variance); %rare kludge to fix bad data
        
        fluxes{i}           = this_flux;
        rest_wavelengths{i} = this_rest_wavelengths;
        
        this_lya_zs = ...
            (this_wavelengths - lya_wavelength) / ...
            lya_wavelength;
        
        % To count the effect of Lyman series from higher z,
        % we compute the absorbers' redshifts for all members of the series
        this_lyseries_zs = nan(numel(this_wavelengths), num_forest_lines);

        for l = 1:num_forest_lines
            this_lyseries_zs(:, l) = ...
              (this_wavelengths - all_transition_wavelengths(l)) / ...
              all_transition_wavelengths(l);
        end

        % DLA existence prior
        less_ind = (prior.z_qsos < (z_qso + prior_z_qso_increase));

        this_num_dlas    = nnz(prior.dla_ind(less_ind));
        this_num_quasars = nnz(less_ind);
        this_p_dla       = this_num_dlas / this_num_quasars;
        this_p_dlas(i)   = this_p_dla;

        %minimal plausible prior to prevent NaN on low z_qso;
        if this_num_dlas == 0
            this_num_dlas = 1;
            this_num_quasars = length(less_ind);
        end
        
        this_sample_log_priors_dla(1, i) = ...
            log(                   this_num_dlas) - log(this_num_quasars);
        this_sample_log_priors_no_dla(1, i) = ...
            log(this_num_quasars - this_num_dlas) - log(this_num_quasars);

        %sample_log_priors_dla(quasar_ind, z_list_ind) = log(.5);
        %sample_log_priors_no_dla(quasar_ind, z_list_ind) = log(.5);

        % fprintf_debug('\n');
        %fprintf_debug(' ...     p(   DLA | z_QSO)        : %0.3f\n',     this_p_dla);
        %fprintf_debug(' ...     p(no DLA | z_QSO)        : %0.3f\n', 1 - this_p_dla);

        % interpolate model onto given wavelengths
        this_mu = mu_interpolator( this_rest_wavelengths);
        this_M  =  M_interpolator({this_rest_wavelengths, 1:k});
        %Debug output
        %all_mus{z_list_ind} = this_mu;
        %all_Ms{z_list_ind} = this_M;

        this_log_omega = log_omega_interpolator(this_rest_wavelengths);
        this_omega2 = exp(2 * this_log_omega);
        
        % Lyman series absorption effect for the noise variance
        % note: this noise variance must be trained on the same number of members of Lyman series
        lya_optical_depth = tau_0 .* (1 + this_lya_zs).^beta;

        % Note: this_wavelengths is within (min_lambda, max_lambda)
        % so it may beyond lya_wavelength, so need an indicator;
        % Note: 1 - exp( -0 ) + c_0 = c_0
        indicator         = this_lya_zs <= z_qso;
        lya_optical_depth = lya_optical_depth .* indicator;

        for l = 2:num_forest_lines
            lyman_1pz = all_transition_wavelengths(1) .* (1 + this_lya_zs) ...
                ./ all_transition_wavelengths(l);

            % only include the Lyman series with absorber redshifts lower than z_qso
            indicator = lyman_1pz <= (1 + z_qso);
            lyman_1pz = lyman_1pz .* indicator;

            tau = tau_0 * all_transition_wavelengths(l) * all_oscillator_strengths(l) ...
                / (  all_transition_wavelengths(1) * all_oscillator_strengths(1) );

            lya_optical_depth = lya_optical_depth + tau .* lyman_1pz.^beta;
        end

        this_scaling_factor = 1 - exp( -lya_optical_depth ) + c_0;
        
        this_omega2 = this_omega2 .* this_scaling_factor.^2;

        % Lyman series absorption effect on the mean-flux
        % apply the lya_absorption after the interpolation because NaN will appear in this_mu
        total_optical_depth = nan(numel(this_wavelengths), num_forest_lines);

        for l = 1:num_forest_lines
            % calculate the oscillator strength for this lyman series member
            this_tau_0 = prev_tau_0 * ...
              all_oscillator_strengths(l)   / lya_oscillator_strength * ...
              all_transition_wavelengths(l) / lya_wavelength;

            total_optical_depth(:, l) = ...
              this_tau_0 .* ( (1 + this_lyseries_zs(:, l)).^prev_beta );

            % indicator function: z absorbers <= z_qso
            % here is different from multi-dla processing script
            % I choose to use zero instead or nan to indicate
            % values outside of the Lyman forest
            indicator = this_lyseries_zs(:, l) <= z_qso;
            total_optical_depth(:, l) = total_optical_depth(:, l) .* indicator;
        end

        % change from nansum to simply sum; shoudn't be different
        % because we also change indicator from nan to zero,
        % but if this script is glitchy then inspect this line
        lya_absorption = exp(- sum(total_optical_depth, 2) );

        this_mu = this_mu .* lya_absorption;
        this_M  = this_M  .* lya_absorption;

        % re-adjust (K + Ω) to the level of μ .* exp( -optical_depth ) = μ .* a_lya
        % now the null model likelihood is:
        % p(y | λ, zqso, v, ω, M_nodla) = N(y; μ .* a_lya, A_lya (K + Ω) A_lya + V)
        this_omega2 = this_omega2 .* lya_absorption.^2;
        
        % baseline: probability of no DLA model
        this_sample_log_likelihoods_no_dla(1, i) = ...
            log_mvnpdf_low_rank(this_flux, this_mu, this_M, ...
            this_omega2 + this_noise_variance);
        %this_sample_log_likelihoods_no_dla(1, i) = this_sample_log_likelihoods_no_dla(1, i) + bw_log_likelihood + rw_log_likelihood;

        %fprintf_debug(' ... log p(D | z_QSO, no DLA)     : %0.2f\n', ...
        %    this_sample_log_likelihoods_no_dla(1, i));

        % Add
        if isempty(this_wavelengths)
            continue;
        end

        % use a temp variable to avoid the possible parfor issue
        % should be fine after change size of min_z_dlas to (num_quasar, num_dla_samples)
        this_min_z_dlas = min_z_dla(this_wavelengths, z_qso);
        this_max_z_dlas = max_z_dla(this_wavelengths, z_qso);

        min_z_dlas(quasar_ind, i) = this_min_z_dlas;
        max_z_dlas(quasar_ind, i) = this_max_z_dlas;

        sample_z_dlas = ...
            this_min_z_dlas +  ...
            (this_max_z_dlas - this_min_z_dlas) * offset_samples;

        used_z_dla(i) = sample_z_dlas(i);

        % ensure enough pixels are on either side for convolving with
        % instrument profile
        padded_wavelengths = ...
            [logspace(log10(min(this_unmasked_wavelengths)) - width * pixel_spacing, ...
            log10(min(this_unmasked_wavelengths)) - pixel_spacing,         ...
            width)';                                                       ...
            this_unmasked_wavelengths;                                              ...
            logspace(log10(max(this_unmasked_wavelengths)) + pixel_spacing,         ...
            log10(max(this_unmasked_wavelengths)) + width * pixel_spacing, ...
            width)'                                                        ...
            ];

        % to retain only unmasked pixels from computed absorption profile
        ind = (~this_pixel_mask(ind));

        % compute probabilities under DLA model for each of the sampled
        % (normalized offset, log(N HI)) pairs
        % absorption corresponding to this sample
        absorption = voigt(padded_wavelengths, sample_z_dlas(i), ...
            nhi_samples(i), num_lines);
        
        % add this line back for implementing pixel masking
        absorption = absorption(ind);

        % delta z = v / c = H(z) d / c = 70 (km/s/Mpc) * sqrt(0.3 * (1+z)^3 + 0.7) * (5 Mpc) / (3x10^5 km/s) ~ 0.005 at z=3
        if add_proximity_zone
            delta_z = (70 * sqrt(.3 * (1+z_qso)^3 + .7) * 5) / (3 * 10^5);
        end


        dla_mu     = this_mu     .* absorption;
        dla_M      = this_M      .* absorption;
        dla_omega2 = this_omega2 .* absorption.^2;
        
        % Add a penalty for short spectra: the expected reduced chi^2 of each spectral pixel that would have been observed.
        this_sample_log_likelihoods_dla(1, i) = ...
            log_mvnpdf_low_rank(this_flux, dla_mu, dla_M, ...
            dla_omega2 + this_noise_variance);
        % Add in penalty for bluewards + redwards spectra
        % this_sample_log_likelihoods_dla(1, i) = this_sample_log_likelihoods_dla(1, i) + bw_log_likelihood + rw_log_likelihood;
    end

    DLA_cut = 20.3;
    sub20pt3_ind = (log_nhi_samples < DLA_cut);

    % indicing sample_log_posteriors instead of assignments to avoid create a new array
    sample_log_posteriors_no_dla(quasar_ind, :) = ...
        this_sample_log_priors_no_dla(1, :) + this_sample_log_likelihoods_no_dla(1, :);
    sample_log_posteriors_dla(quasar_ind, :)    = ...
        this_sample_log_priors_dla(1, :)    + this_sample_log_likelihoods_dla(1, :);
    sample_log_posteriors_dla_sub(quasar_ind, :) = this_sample_log_priors_dla(1, sub20pt3_ind) + this_sample_log_likelihoods_dla(1, sub20pt3_ind);
    sample_log_posteriors_dla_sup(quasar_ind, :) = this_sample_log_priors_dla(1, ~sub20pt3_ind) + this_sample_log_likelihoods_dla(1, ~sub20pt3_ind);

    % use nanmax to avoid NaN potentially in the samples
    % not sure whether the z code has many NaNs in array; the multi-dla codes have many NaNs
    max_log_likelihood_no_dla = nanmax(sample_log_posteriors_no_dla(quasar_ind, :));
    max_log_likelihood_dla    = nanmax(sample_log_posteriors_dla(quasar_ind, :));
    max_log_likelihood_dla_sub = nanmax(sample_log_posteriors_dla_sub(quasar_ind, :));
    max_log_likelihood_dla_sup = nanmax(sample_log_posteriors_dla_sup(quasar_ind, :));

    probabilities_no_dla = exp(sample_log_posteriors_no_dla - max_log_likelihood_no_dla);
    probabilities_dla = exp(sample_log_posteriors_dla - max_log_likelihood_dla);
    probabilities_dla_sub = exp(sample_log_posteriors_dla_sub - max_log_likelihood_dla_sub);
    probabilities_dla_sup = exp(sample_log_posteriors_dla_sup - max_log_likelihood_dla_sup);

    probabilities_no_dla = ...
        exp(sample_log_posteriors_no_dla(quasar_ind, :) - ...
            max_log_likelihood_no_dla);
    probabilities_dla    = ... 
        exp(sample_log_posteriors_dla(quasar_ind, :) - ...
            max_log_likelihood_dla);
    probabilities_dla_sub = exp(sample_log_posteriors_dla_sub(quasar_ind, :) - max_log_likelihood_dla_sub);
    probabilities_dla_sup = exp(sample_log_posteriors_dla_sup(quasar_ind, :) - max_log_likelihood_dla_sup);

    [~, I] = nanmax(probabilities_no_dla + probabilities_dla);

    z_map(quasar_ind) = offset_samples_qso(I);                                  %MAP estimate

    [~, I] = nanmax(probabilities_dla);
    z_dla_map(quasar_ind) = used_z_dla(I);
    n_hi_map(quasar_ind) = nhi_samples(I);

    log_posteriors_no_dla(quasar_ind) = log(mean(probabilities_no_dla)) + max_log_likelihood_no_dla;   %Expected
    log_posteriors_dla(quasar_ind) = log(mean(probabilities_dla)) + max_log_likelihood_dla;            %Expected
    log_posteriors_dla_sub(quasar_ind) = log(mean(probabilities_dla_sub)) + max_log_likelihood_dla_sub;            %Expected
    log_posteriors_dla_sup(quasar_ind) = log(mean(probabilities_dla_sup)) + max_log_likelihood_dla_sup;            %Expected

    %fprintf_debug(' ... log p(D | z_QSO,    DLA)     : %0.2f\n', ...
    %    log_likelihoods_dla(quasar_ind));
    %fprintf_debug(' ... log p(DLA | D, z_QSO)        : %0.2f\n', ...
    %    log_posteriors_dla(quasar_ind));

    fprintf(' took %0.3fs.\n', toc);
%    if mod(quasar_ind, 50) == 0
%        save(['./checkpointing/curDLA_', optTagFull, '.mat'], 'log_posteriors_dla_sub', 'log_posteriors_dla_sup', 'log_posteriors_dla', 'log_posteriors_no_dla', 'z_true', 'dla_true', 'quasar_ind', 'quasar_num',...
%            'sample_log_likelihoods_dla', 'sample_log_likelihoods_no_dla', 'sample_z_dlas', 'nhi_samples', 'offset_samples_qso', 'offset_samples', 'z_map', 'signal_to_noise', 'z_dla_map', 'n_hi_map');
%    end
end


% compute model posteriors in numerically safe manner
max_log_posteriors = ...
    max([log_posteriors_no_dla, log_posteriors_dla_sub, log_posteriors_dla_sup], [], 2);

model_posteriors = ...
    exp([log_posteriors_no_dla,  log_posteriors_dla_sub, log_posteriors_dla_sup] - max_log_posteriors);

model_posteriors = model_posteriors ./ sum(model_posteriors, 2);

p_no_dlas = model_posteriors(:, 1);
p_dlas    = 1 - p_no_dlas;

% save results
variables_to_save = {'training_release', 'training_set_name', ...
    'dla_catalog_name', 'release', ...
    'test_set_name', 'test_ind', 'prior_z_qso_increase', ...
    'max_z_cut', 'num_lines', ... %'min_z_dlas', 'max_z_dlas', ... % save memory
    'sample_log_posteriors_no_dla', 'sample_log_posteriors_dla', ...
    'log_posteriors_no_dla', 'log_posteriors_dla', ...
    'model_posteriors', 'p_no_dlas', ...
    'p_dlas', 'z_map', 'z_true', 'dla_true', 'z_dla_map', 'n_hi_map', 'signal_to_noise', 'all_thing_ids'};

    % 'sample_log_priors_no_dla', 'sample_log_priors_dla', ...
    % 'sample_log_likelihoods_no_dla', 'sample_log_likelihoods_dla', ...

filename = sprintf('%s/processed_qsos_%s-%s_%d-%d', ...
    processed_directory(release), ...
    test_set_name, optTag, ...
    qso_ind(1), qso_ind(1) + numel(qso_ind));

save(filename, variables_to_save{:}, '-v7.3');

