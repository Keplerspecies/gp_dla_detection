function [] = read_data
    clear
    clc

    dataPrep('full/table2.dat', 'full/table1.dat', 'data.mat');
    load data.mat %d(1)=ID, d(2)=DLA log odds, d(3)=notDLA, d(4)=QSO present
    
    logodds = d(:,2);
    o = ones(size(logodds));
    p = exp(logodds)./(exp(logodds)+o);
    
    ps = [];
    for i = 0:.05:1
        po = find(p < i);
        ps = [ps, length(po)];
    end
    
    ps = ps./length(logodds);
    fp = ones(size(ps))-ps;
    
    plot(fp, ps);
end

function [] = dataPrep(bfName, qsoName, saveName)
    [bfMap, bfnMap] = BFGet(bfName);
    qsoMap = isQSO(qsoName);
    %save(saveName, 'bfMap', 'qsoMap');
    %load(saveName);
    
    bfKeys = keys(bfMap);
    count = 0;
    d = [];
    for i = 1:length(bfKeys)
        bfk = bfKeys(i);
        bfk = bfk{1};
        bf = bfMap(bfk);
        bfn = bfnMap(bfk);
        if isKey(qsoMap, bfk)
            count = count + 1;
            qso = qsoMap(bfk);
            qsoTrip = [str2num(bfk), bf, bfn, qso];
            d(count, :) = qsoTrip;
        end
    end
    save(saveName, 'bfMap', 'bfnMap', 'qsoMap', 'd');
end

function [qsoMap] = isQSO(dat)
    row = 11;
    fname = ['./data/dr12q/processed/dat/', dat];
    fid = fopen(fname, 'r');
    
    res = textscan(fid, '%f');
    res = res{1};
    res = reshape(res, row, floor(length(res)/row));
    size(res)
    qsoKey = res(1,:);
    qso = res(11,:);
    
    qsoMap = containers.Map;
    for i=1:length(qso)
       qsoa = num2str(qso(i), '%04d');
       qsoa = qsoa(2);
       qsoMap(num2str(qsoKey(i))) = str2num(qsoa);
    end
end

function [bfMap, bfnMap] = BFGet(dat)
    row = 11;
    fname = ['./data/dr12q/processed/dat/', dat];
    fid = fopen(fname, 'r');
    
    res = textscan(fid, '%f');
    res = res{1};
    res = reshape(res, row, floor(length(res)/row));
    size(res)
    bfKey = res(1, :);
    logNotDLA = res(6, :);
    logDLA = res(7, :);
    logPriorNotDLA = res(4, :);
    logPriorDLA = res(5, :);
    
    priorDLA = exp(logPriorDLA);
    priorDLAOdds = priorDLA./(ones(size(priorDLA))-priorDLA);
    logPriorDLAOdds = log(priorDLAOdds);
    
    bf = logDLA-logNotDLA;
    bfn = logDLA-logNotDLA; %repair if needed
    
    
    bfMap = containers.Map;
    bfnMap = containers.Map;
    for i=1:length(bfKey) 
       bfMap(num2str(bfKey(i))) = bf(i);
       bfnMap(num2str(bfKey(i))) = bfn(i);
    end
end