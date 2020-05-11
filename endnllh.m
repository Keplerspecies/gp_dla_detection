function nllh = endnllh(obs,obsnoisevars,endsigma2,endmu)

	dens = obsnoisevars + endsigma2;

	if nargin<4
		endmu = fitendmu(endsigma2,obs,obsnoisevars,dens);
	end

	diffs = obs-endmu;

	nllh = length(obs)*log(sqrt(2*pi)) ...
		+ sum(log(dens) + diffs.*diffs./dens)/2;
