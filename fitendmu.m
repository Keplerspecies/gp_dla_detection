function endmu = fitendmu(endsigma2,obs,obsnoisevars,precompdens)

	if nargin<4
		precompdens = obsnoisevars + endsigma2;
	end

	endmu = sum(obs./precompdens)/sum(1.0./precompdens);
