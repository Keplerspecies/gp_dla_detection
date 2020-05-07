disp('starting...');
preqsos = matfile('/data/jfaub001/quasar/gp_dla_detection-zqsos2/data/dr12q/processed/oldpreloaded_qsos.mat');
load('./test/train_ind.mat');
all_flux           =                 preqsos.all_flux;
all_flux           =           all_flux(train_ind, :);
all_noise_variance =       preqsos.all_noise_variance;
all_noise_variance = all_noise_variance(train_ind, :);

f_s = ''; nv_s = ''; nl = newline;
count = 0; i = 0;
for i=1:length(all_flux)
%while count ~= 100
    if mod(i, 100) == 0
	disp(i);
	disp(length(all_flux));
    end

    if mod(i, 500) == 0
        disp('saving...');
        save('fnv.mat', 'f_s', 'nv_s', 'i');
    end
    i = i + 1;
    f_l = ''; nv_l = '';
    f = all_flux{i};
    nv = all_noise_variance{i};
    if isempty(f)
        continue;
    end
    count = count + 1;
    f_l = sprintf('%f,', f);
    nv_l = sprintf('%f,', nv);
    %for j=1:length(f)
    %    f_l = [f_l, ', ', num2str(f(j))];
    %    nv_l = [nv_l, ', ', num2str(nv(j))];
    %end
    f_s = [f_s, f_l, nl];
    nv_s = [nv_s, nv_l, nl];
end
f_id = fopen('f.txt', 'w');
nv_id = fopen('nv.txt', 'w');
fprintf(f_id, '%s', f_s);
fprintf(nv_id, '%s', nv_s);

