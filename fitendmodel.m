function [endmu,endsigma2] = fitendmodel(obs,obsnoisevars)

	touse = isfinite(obsnoisevars);
	obs = obs(touse);
	obsnoisevars = obsnoisevars(touse);

	naivemu = sum(obs./obsnoisevars)/sum(1.0./obsnoisevars);
	diffs = obs-naivemu;
	naivesigma2 = sum(diffs.*diffs./obsnoisevars)/sum(1.0./obsnoisevars);

	spread = 100;

	[endsigma2,nllh,eflag,out] = fminbnd(@(s2) endnllh(obs,obsnoisevars,s2),...
							naivesigma2/spread,naivesigma2*spread);
	if eflag <= 0
		display("end model failed to converge");
		endsigma2 = naivesigma2;
	end

	endmu = fitendmu(endsigma2,obs,obsnoisevars);
