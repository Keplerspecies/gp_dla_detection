% process_qsos: run DLA detection algorithm on specified objects
% 
% Apr 8, 2020: add all Lyman series to the effective optical depth
%   effective_optical_depth := ∑ τ fi1 λi1 / ( f21 λ21 ) * ( 1 + z_i1 )^β
%  where 
%   1 + z_i1 =  λobs / λ_i1 = λ_lya / λ_i1 *  (1 + z_a)
% Dec 25, 2019: add Lyman series to the noise variance training
%   s(z)     = 1 - exp(-effective_optical_depth) + c_0 
% the mean values of Kim's effective optical depth
%
% Apr 28: add occams razor for penalising the missing pixels,
%   this factor is tuned to affect log likelihood in a range +- 500,
%   this value could be effective to penalise every likelihoods for zQSO > zCIV
%   the current implemetation is:
%     likelihood - occams_factor * (1 - lambda_observed / (max_lambda - min_lambda) )
%   and occams_factor is a tunable hyperparameter
prev_tau_0 = 0.0023;
prev_beta  = 3.65;

occams_factor = 1000;

% load QSO model from training release
variables_to_load = {'rest_wavelengths', 'mu', 'M'};
load(sprintf('%s/learned_zqso_only_model_%s_norm_%d-%d',             ...
    processed_directory(training_release), ...
    training_set_name, ...
    normalization_min_lambda, normalization_max_lambda),                    ...
    variables_to_load{:});

% load redshifts from catalog to process
catalog = load(sprintf('%s/zqso_only_catalog', processed_directory(release)));

z_qsos = catalog.z_qsos;

rng('default');
sequence = scramble(haltonset(1), 'rr2');

% ADDING: second dimension for z_qso
offset_samples_qso  = sequence(1:num_zqso_samples, 1)';

bins = 150;
[z_freq, z_bin] = histcounts(z_qsos, z_qso_cut : ((max(z_qsos) - z_qso_cut) / bins) : max(z_qsos));
for i=length(z_freq):-1:1
    z_freq(i) = sum(z_freq(1:i));
end

z_freq = [0 z_freq];
z_freq = z_freq / max(z_freq);
[z_freq, I] = unique(z_freq);
z_bin = z_bin(I);

offset_samples_qso = interp1(z_freq, z_bin, offset_samples_qso);

% load preprocessed QSOs
variables_to_load = {'all_wavelengths', 'all_flux', 'all_noise_variance', ...
    'all_pixel_mask'};
load(sprintf('%s/preloaded_zqso_only_qsos', processed_directory(release)), ...
    variables_to_load{:});
test_ind = (catalog.filter_flags == 0);
all_wavelengths    =    all_wavelengths(test_ind);
all_flux           =           all_flux(test_ind);
all_noise_variance = all_noise_variance(test_ind);
all_pixel_mask     =     all_pixel_mask(test_ind);
all_thing_ids      =   catalog.thing_ids(test_ind);

z_qsos = catalog.z_qsos(test_ind);

num_quasars = numel(z_qsos);
if exist('qso_ind', 'var') == 0
    qso_ind = 1:1:floor(num_quasars/100);
end
num_quasars = numel(qso_ind);

%load('./test/M.mat');
% preprocess model interpolants
mu_interpolator = ...
    griddedInterpolant(rest_wavelengths,        mu,        'linear');
M_interpolator = ...
    griddedInterpolant({rest_wavelengths, 1:k}, M,         'linear');

% initialize results
sample_log_posteriors  = nan(num_quasars, num_zqso_samples);
z_true                        = nan(num_quasars, 1);
z_map                         = nan(num_quasars, 1);
signal_to_noise               = nan(num_quasars, 1);

z_list                   = 1:length(offset_samples_qso);
%Debug output
%all_mus = cell(size(z_list));

fluxes                   = cell(length(z_list), 1);
rest_wavelengths         = cell(length(z_list), 1);

% this is just an array allow you to select a range
% of quasars to run
quasar_ind = 1;
q_ind_start = quasar_ind;

% catch the exceptions
all_exceptions = false(num_quasars, 1);

for quasar_ind = q_ind_start:num_quasars %quasar list
    tic;
    quasar_num = qso_ind(quasar_ind);
    z_true(quasar_ind)   = z_qsos(quasar_num);

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

    this_noise_variance(isinf(this_noise_variance)) = .01; %kludge to fix bad data
    
    this_pixel_signal_to_noise  = sqrt(this_noise_variance) ./ abs(this_flux);

    % this is before pixel masking; nanmean to avoid possible NaN values
    signal_to_noise(quasar_ind) = nanmean(this_pixel_signal_to_noise);

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
    
    parfor i = 1:num_zqso_samples       %variant redshift in quasars
        z_qso = offset_samples_qso(i);

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
        
        % if (min_observed_lambda > max_observed_lambda) | (nnz(ind) < 150)
        %     % If we have no data in the observed range, this sample is maximally unlikely.
        %     sample_log_posteriors(quasar_ind, i) = -1.e50;
        %     continue;
        % end
        %ind = ind & (~this_pixel_mask);
        
        ind = ind & (~this_pixel_mask);

        this_wavelengths      =      this_wavelengths(ind);
        this_rest_wavelengths = this_rest_wavelengths(ind);
        this_flux             =             this_flux(ind);
        this_noise_variance   =   this_noise_variance(ind);

        this_noise_variance(isinf(this_noise_variance)) = nanmean(this_noise_variance); %rare kludge to fix bad data
        
        fluxes{i}           = this_flux;
        rest_wavelengths{i} = this_rest_wavelengths;
        
        % interpolate model onto given wavelengths
        this_mu = mu_interpolator( this_rest_wavelengths);
        this_M  =  M_interpolator({this_rest_wavelengths, 1:k});
        %Debug output
        %all_mus{z_list_ind} = this_mu;
        %all_Ms{z_list_ind} = this_M;
       
        sample_log_priors = 0;

        % additional occams razor for penalizing the not enough data points in the window
        occams = occams_factor * (1 - lambda_observed / (max_lambda - min_lambda) );

        sample_log_posteriors(quasar_ind, i) = ...
            log_mvnpdf_low_rank(this_flux, this_mu, this_M, this_noise_variance) + sample_log_priors ...
            - occams;

        % % Correct for incomplete data
        % corr = nnz(ind) - length(this_rest_wavelengths);
        % sample_log_posteriors(quasar_ind, i) = sample_log_posteriors(quasar_ind, i) + corr;

        % fprintf_debug(' ... log p(D | z_QSO)     : %0.2f\n', ...
        %     sample_log_posteriors(quasar_ind, i));
    end
    this_sample_log = sample_log_posteriors(quasar_ind, :);
    
    [~, I] = nanmax(this_sample_log);
    
    z_map(quasar_ind) = offset_samples_qso(I);                                  %MAP estimate

    fprintf(' took %0.3fs.\n', toc);

    zdiff = z_map(quasar_ind) - z_qsos(quasar_ind);
    if mod(quasar_ind, 1) == 0
        t = toc;
        fprintf('Done QSO %i of %i in %0.3f s. True z_QSO = %0.4f, I=%d map=%0.4f dif = %.04f\n', ...
            quasar_ind, num_quasars, t, z_qsos(quasar_ind), I, z_map(quasar_ind), zdiff);
    end
end

% save results
variables_to_save = {'training_release', 'training_set_name', 'offset_samples_qso', 'sample_log_posteriors', ...
     'z_map', 'z_qsos', 'all_thing_ids', 'test_ind', 'z_true'};

filename = sprintf('%s/processed_zqso_only_qsos_%s-%s_%d-%d_%d-%d_oc%d', ...
    processed_directory(release), ...
    test_set_name, optTag, ...
    qso_ind(1), qso_ind(1) + numel(qso_ind), ...
    normalization_min_lambda, normalization_max_lambda, occams_factor);

save(filename, variables_to_save{:}, '-v7.3');
