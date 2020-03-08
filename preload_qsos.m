% preload_qsos: loads spectra from SDSS FITS files, applies further
% filters, and applies some basic preprocessing such as normalization
% and truncation to the region of interest

% load QSO catalog
variables_to_load = {'z_qsos', 'plates', 'mjds', 'fiber_ids', 'filter_flags'};
load(sprintf('%s/catalog', processed_directory(release)), ...
    variables_to_load{:});

num_quasars = numel(z_qsos);

all_wavelengths    =  cell(num_quasars, 1);
all_flux           =  cell(num_quasars, 1);
all_noise_variance =  cell(num_quasars, 1);
all_pixel_mask     =  cell(num_quasars, 1);
all_normalizers    = zeros(num_quasars, 1);

for i = 1:num_quasars
  if (filter_flags(i) > 0)
    continue;
  end

  [this_wavelengths, this_flux, this_noise_variance, this_pixel_mask] ...
      = file_loader(plates(i), mjds(i), fiber_ids(i));

  % normalize flux

  ind = (this_wavelengths >= normalization_min_lambda * (z_qso_cut) + 1) & ... // min allowable in observer frame
        (this_wavelengths <= normalization_max_lambda * (z_qso_training_max_cut) + 1) & ... //max allowable in observer frame
        (~this_pixel_mask);
  medfiltwidth = (normalization_max_lambda - normalization_min_lambda)/dlambda;
  this_norm = nanmax(medfilt1(this_flux(ind), medfiltwidth, 'omitnan', 'truncate'));

  %{
  this_rest_wavelengths = emitted_wavelengths(this_wavelengths, z_qsos(i));
  ind = (this_rest_wavelengths >= normalization_min_lambda) & ... // min possible in observer frame
        (this_rest_wavelengths <= normalization_max_lambda) & ...
        (~this_pixel_mask);
  this_norm = nanmedian(this_flux(ind));
  this_hold(i) = min(this_rest_wavelengths);
  %}

  % bit 2: cannot normalize (all normalizing pixels are masked)
  if (isnan(this_norm))
    filter_flags(i) = bitset(filter_flags(i), 3, true);
    continue;
  end


  ind = (this_wavelengths >= (min_lambda * (z_qso_cut) + 1)) & ...
        (this_wavelengths <= (max_lambda * (z_qso_training_max_cut) + 1)) & ...
        (~this_pixel_mask);
  %{
  ind = (this_rest_wavelengths >= min_lambda) & ...
        (this_rest_wavelengths <= max_lambda) & ...
        (~this_pixel_mask);
  %}
  % bit 3: not enough pixels available
  if (nnz(ind) < min_num_pixels)
    filter_flags(i) = bitset(filter_flags(i), 4, true);
    continue;
  end

  all_normalizers(i) = this_norm;

  this_flux           = this_flux           / this_norm;
  this_noise_variance = this_noise_variance / this_norm^2;


  ind = (this_wavelengths >= (loading_min_lambda * (z_qso_cut) + 1)) & ...
        (this_wavelengths <= (loading_max_lambda * (z_qso_training_max_cut) + 1));

  %{
  ind = (this_rest_wavelengths >= loading_min_lambda) & ...
        (this_rest_wavelengths <= loading_max_lambda);
  %}
  % add one pixel on either side
  available_ind = find(~ind & ~this_pixel_mask);
  ind(min(available_ind(available_ind > find(ind, 1, 'last' )))) = true;
  ind(max(available_ind(available_ind < find(ind, 1, 'first')))) = true;

  all_wavelengths{i}    =    this_wavelengths;%no longer (ind)
  all_flux{i}           =           this_flux;
  all_noise_variance{i} = this_noise_variance;
  all_pixel_mask{i}     =     this_pixel_mask;

  fprintf('loaded quasar %i of %i (%i/%i/%04i)\n', ...
          i, num_quasars, plates(i), mjds(i), fiber_ids(i));
end

variables_to_save = {'loading_min_lambda', 'loading_max_lambda', ...
                     'normalization_min_lambda', 'normalization_max_lambda', ...
                     'min_num_pixels', 'all_wavelengths', 'all_flux', ...
                     'all_noise_variance', 'all_pixel_mask', ...
                     'all_normalizers'};
save(sprintf('%s/preloaded_qsos', processed_directory(release)), ...
     variables_to_save{:}, '-v7.3');

% write new filter flags to catalog
save(sprintf('%s/catalog', processed_directory(release)), ...
     'filter_flags', '-append');
