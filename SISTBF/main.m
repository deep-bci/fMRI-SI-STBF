clc
clear
addpath('.\support')
addpath('.\SISTBF')
load('.\Gain.mat'); % Lead-filed matrix
load('.\Cortex.mat'); % Cortex
load('.\GridLoc.mat'); % GridLoc
load('.\pQ.mat'); % fmri-informed clusters of sub 01 in multimodal face-processing dataset
load('.\invalid_pQ.mat'); % invalid clausters by reflecting the original SPM{T} image in the y? and z?directions
load('.\Average_EEGdata.mat'); % Average EEG data (famous + unfamiliar) of sub 01 in multimodal face-processing dataset
algorithms = {'fMRI-SI-STBF'};%
DDPScale = 4;   % DDP scale for construct EEG-informed clusters
[~,~,V] = svd(Data_average,'econ');
pst = -200 :2 : 800; % time points (ms)
%%
fmri_prior = [pQ([7 8]), invalid_pQ([10 12 4])]; % fMRI priors for source reconstruction, where the first two priors are valid fMRI priors, and the last three are invalid priros

OPTIONS.uniform       = 1;
OPTIONS.WGN           = 1;
OPTIONS.ar            = 0;
OPTIONS.Amp           = 1e-4;
OPTIONS.SNR           = 5;
OPTIONS.pQ            = pQ([7 8 9]);  % clusters for simulated EEG data 
OPTIONS.TBFs          = V(:,1:3)';
OPTIONS.MixMatrix     = eye(3);

[Sim_Data,s_real,Result] = Simulation_Data_Generate_fmri_prior (LFM_Primary,Cortex,pst,OPTIONS);
ActiveVoxSeed = Result.ActiveVoxSeed;
fprintf('Actual SNR is %g\n',20*log10(norm(LFM_Primary*s_real,'fro')/norm(Sim_Data-LFM_Primary*s_real,'fro')));

StimTime = find(abs(pst) == min(abs(pst)));
Cov_n =  Sim_Data(:,1:StimTime)*Sim_Data(:,1:StimTime)'/(StimTime - 1);
%% Reducing the leadfield matrix
u = 1;%spm_svd(LFM_Primary*LFM_Primary');
LFM_reduce = u'*LFM_Primary;
Sim_Data = u'*Sim_Data;
Cov_n = u'*Cov_n*u;
clear u
%% Data scale
Scale = 0;
ratio = 1;
B = Sim_Data;
LFM_scale = LFM_reduce;
Cov_n = Cov_n./ratio.^2;
 %% Whiten the measurements and leadfield matrix  
rnkC_noise = rank(single(Cov_n));
variance = diag(Cov_n);
isPca = 1;
if isequal(Cov_n, diag(variance))
    isPca = 0;
end
[V,D] = eig(Cov_n);
D = diag(D);
[D,I] = sort(D,'descend');
V = V(:,I);
if ~isPca
    D = 1./D;
    W = diag(sqrt(D)) * V';
else
    D = 1 ./ D;
    D(rnkC_noise+1:end) = 0;
    W = diag(sqrt(D)) * V';
    W = W(1:rnkC_noise,:);
end
clear V D I
LFM = W*LFM_scale;
B = W*B;
clear W LFM_scale; 
%% fMRI-SI-STBF
if any(strcmpi('fMRI-SI-STBF', algorithms))
    tic
    [SBFs,seed,cluster] = SBFConstruct(B,LFM,Cortex.VertConn,DDPScale,'fmriprior',fmri_prior);
    cls = zeros(numel(cluster),1);
    for i = 1:numel(cluster)
        cls(i) = numel(cluster{i});
    end
    [S_fMRI_STBF,par_fMRI_STBF] = STBFSI(B,LFM,SBFs,'cls',cls,'epsilon',1e-6,'max_iter',500,'prune',[1e-6,1e-1],'ifplot',0,'kinitial',10);
     S_fMRI_STBF = S_fMRI_STBF*ratio;
     toc
    Result.fMRI_STBF.S = S_fMRI_STBF;
    Result.fMRI_STBF.par = par_fMRI_STBF;
end






