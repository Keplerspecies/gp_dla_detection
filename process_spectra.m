function [] = process_spectra(start_, end_, inc)
    %clear
    %clc 

    %load('/home/chaki/Research/quasar/gp_dla_detection/data/dr12q/processed/processed_qsos_dr12q.mat');
    %genroc(prior_catalog, test_ind)
    %find distribution of min lambda for min z redshit across quasars in
    %training, graph redshifts of DLA occurences
    
    set_parameters;
    training_release  = 'dr12q';
prior_catalog = ...
    load(sprintf('%s/catalog', processed_directory(training_release)));
    load('./test/process_results.mat');

    for i = start_:end_
    optTagFull = [optTag, '-', num2str(i)];
    %inc = 50000;
    qso_ind = [inc * (i-1) + 1:inc*i]; %indices in test set to iterate over
    
    filter(); %switch to array of values
    spectra_setup();
    ps([1:sum(test_ind(qso_ind))], optTagFull);
    end
    %gen_diag(list(qso_ind(1)));
end

function [auc] = genroc(prior_catalog, test_ind)
    model_posteriors = prior_catalog.model_posteriors(:,2)';
    ground_truth = prior_catalog.dla_inds('dr12q_visual'); %actual truth unknown, taken for ground
    ground_truth_tested = ground_truth(test_ind)';
    in_dr10_tested = prior_catalog.in_dr10(test_ind);
    
    [tpr, fpr, thresholds, auc] = perfcurve(ground_truth_tested, model_posteriors, 1);
    plotroc(ground_truth_tested, model_posteriors);
end

function [] = ps(qso_ind, optTagFull)
% produce catalog searching [Lyoo + 3000 km/s, Lya - 3000 km/s]
set_parameters;

% specify the learned quasar model to use
training_release  = 'dr12q';
training_set_name = 'dr9q_minus_concordance';

% specify the spectra to use for computing the DLA existence prior
dla_catalog_name  = 'dr9q_concordance';
prior_ind = ...
    [' prior_catalog.in_dr9 & '             ...
     '(prior_catalog.filter_flags == 0) & ' ...
     ' prior_catalog.los_inds(dla_catalog_name)'];

% specify the spectra to process
release = 'dr12q'; 
test_set_name = 'dr12q';
test_ind = '(catalog.filter_flags == 0)';

optTagFull = [release, '-', optTagFull];

% testing: repeat for safety
generate_dla_samples;

% process the spectra
if integrate
    playground3;
else
    process_qsos;
end
end

function [] = spectra_presetup()
set_parameters;
release = 'dr12q';

file_loader = @(plate, mjd, fiber_id) ...
  (read_spec(sprintf('%s/%i/spec-%i-%i-%04i.fits', ...
    spectra_directory(release),                  ...
    plate,                                       ...
    plate,                                       ...
    mjd,                                         ...
    fiber_id)));

preload_qsos;
end

function [] = spectra_setup()
set_parameters;

training_release  = 'dr12q';
dla_catalog_name = 'dr9q_concordance';
train_ind = ...
    [' catalog.in_dr9                     & ' ...
     '(catalog.filter_flags == 0)         & ' ...
     ' catalog.los_inds(dla_catalog_name) & ' ...
     '~catalog.dla_inds(dla_catalog_name)'];

cd minFunc_2012
addpath(genpath(pwd));
mexAll;
% mex -v CFLAGS="\$CFLAGS -std=c99 -I/home/csgrads/jfaub001/Research/libcerf/include -L/home/csgrads/jfaub001/Research/libcerf/lib voigt.c -lcerfexport
% setenv('LD_RUN_PATH', '/home/csgrads/jfaub001/Research/libcerf/lib')
% maybe LD_LIBRARY_PATH as well
cd ..

training_set_name = 'dr9q_minus_concordance';
learn_qso_model;
training_release  = 'dr12q';
%generate_dla_samples;

end

%include index of all numbers to be used; e.g. [1:500] uses the first 500
%quasars
function [] = filter(varargin)
    set_parameters;
    build_catalogs;
    
    if numel(varargin) < 1
        START = 1;
        END = numel(z_qsos);
        IND = START:1:END;
    else
        IND = varargin{1};
    end
    load([base_directory, '/dr12q/processed/oldcatalog.mat']);

    tic;
    
    dla_inds = filter_helper(dla_inds, IND);
    log_nhis = filter_helper(log_nhis, IND);
    los_inds = filter_helper(los_inds, IND);
    z_dlas = filter_helper(z_dlas, IND);
    
    bal_visual_flags = bal_visual_flags(IND);
    decs = decs(IND);
    fiber_ids = fiber_ids(IND);
    filter_flags = filter_flags(IND);
    in_dr10 = in_dr10(IND);
    in_dr9 = in_dr9(IND);
    mjds = mjds(IND);
    plates = plates(IND);
    ras = ras(IND);
    sdss_names = sdss_names(IND);
    snrs = snrs(IND);
    thing_ids = thing_ids(IND);
    z_qsos = z_qsos(IND);
    save([base_directory, '/dr12q/processed/catalog.mat']);
    clear('dla_inds', 'log_nhis', 'los_inds', 'z_dlas', 'bal_visual_flags', 'decs', ...
        'fiber_ids', 'filter_flags', 'in_dr10', 'in_dr9', 'mjds', 'plates', 'ras', ...
        'sdss_names', 'snrs', 'thing_ids', 'z_qsos');
    
    if length(IND) > 1000000
        preqsos = matfile([base_directory, '/dr12q/processed/oldpreloaded_qsos.mat']);
        all_flux = preqsos.all_flux(IND, 1);
        all_noise_variance = preqsos.all_noise_variance(IND, 1);
        all_normalizers = preqsos.all_normalizers(IND, 1);
        all_pixel_mask = preqsos.all_pixel_mask(IND, 1);
        all_wavelengths = preqsos.all_wavelengths(IND, 1);
        save([base_directory, '/dr12q/processed/preloaded_qsos.mat'], '-v7.3');
    else
        spectra_presetup();
    end
    disp(toc);
    %save([base_directory, '/dr12q/processed/oldcatalog.mat']);
end

function [lmap] = filter_helper(map, IND)
    k = keys(map);
    for i = k
        i = i{1};
        v = map(i);
        v = v(IND);
        map(i) = v;
    end
    lmap = map;
end
