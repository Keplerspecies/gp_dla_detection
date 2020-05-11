% set_parameters: sets various parameters for the DLA detection
% pipeline

%flags for changes
extrapolate_subdla = 0; %0 = off, 1 = on
add_proximity_zone = 0;
integrate          = 1;
optTag = [num2str(integrate), num2str(extrapolate_subdla), num2str(add_proximity_zone)];

% physical constants
lya_wavelength = 1215.6701;                   % Lyman alpha transition wavelength  Å
lyb_wavelength = 1025.7223;                   % Lyman beta  transition wavelength  Å
lyman_limit    =  911.7633;                   % Lyman limit wavelength             Å
speed_of_light = 299792458;                   % speed of light                     m s⁻¹

% converts relative velocity in km s^-1 to redshift difference
kms_to_z = @(kms) (kms * 1000) / speed_of_light;

% utility functions for redshifting
emitted_wavelengths = ...
    @(observed_wavelengths, z) (observed_wavelengths / (1 + z));

observed_wavelengths = ...
    @(emitted_wavelengths,  z) ( emitted_wavelengths * (1 + z));

release = 'dr12q';
file_loader = @(plate, mjd, fiber_id) ...
  (read_spec(sprintf('%s/%i/spec-%i-%i-%04i.fits', ...
    spectra_directory(release),                  ...
    plate,                                       ...
    plate,                                       ...
    mjd,                                         ...
    fiber_id)));

training_release  = 'dr12q';
training_set_name = 'dr9q_minus_concordance';
train_ind = ...
    [' catalog.in_dr9                     & ' ...
     '(catalog.filter_flags == 0) ' ];

test_set_name = 'dr12q';

% file loading parameters
loading_min_lambda = lya_wavelength;                % range of rest wavelengths to load  Å
loading_max_lambda = 5000;                  % This maximum is set so we include CIV.
% The maximum allowed is set so that even if the peak is redshifted off the end, the
% quasar still has data in the range

% preprocessing parameters
z_qso_cut      = 2.15;         % filter out QSOs with z less than this threshold
z_qso_training_max_cut = 5; % roughly 95% of training data occurs before this redshift; assuming for normalization purposes (move to set_parameters when pleased)
z_qso_training_min_cut = 1.5; % Ignore these quasars when training
min_num_pixels = 400;                         % minimum number of non-masked pixels

% normalization parameters
% I use 1216 is basically because I want integer in my saved filenames
normalization_min_lambda = 1216 - 40;              % range of rest wavelengths to use   Å
normalization_max_lambda = 1216 + 40;              %   for flux normalization

% null model parameters
min_lambda         = lya_wavelength - 40;                 % range of rest wavelengths to       Å
max_lambda         = 3000;                 %   model
dlambda            = 0.25;                 % separation of wavelength grid      Å
k                  = 20;                      % rank of non-diagonal contribution
max_noise_variance = 4^2;                     % maximum pixel noise allowed during model training

% optimization parameters
minFunc_options =               ...           % optimization options for model fitting
    struct('MaxIter',     4000, ...
           'MaxFunEvals', 8000);

num_zqso_samples     = 10000;                  % number of parameter samples

% base directory for all data
base_directory = 'data';

% utility functions for identifying various directories
distfiles_directory = @(release) ...
    sprintf('%s/%s/distfiles', base_directory, release);

spectra_directory   = @(release) ...
    sprintf('%s/%s/spectra',   base_directory, release);

processed_directory = @(release) ...
    sprintf('%s/%s/processed', base_directory, release);

% replace with @(varargin) (fprintf(varargin{:})) to show debug statements
% fprintf_debug = @(varargin) (fprintf(varargin{:}));
fprintf_debug = @(varargin) ([]);
