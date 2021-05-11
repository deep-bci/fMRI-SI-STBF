function [B,s_real,Result] = Simulation_Data_Generate_fmri_prior (LFM,Cortex,Time,OPTIONS)
% Descriptions: Genarate simulated EEG/MEG data for extended sources using
% on fMRI prior
% Inputs: LFM: Lead field Matrix(#sensors X #sources)
%         Cortex.| Vertices
%               .| Faces    Cortex files
%         Time: Time for each sample
%         OPTIONS. DefinedArea: Areas for each patch
%                . seedvox    : seedvoxs of each patch
%                . frequency  : frequency of the gaussian damped time courses
%                . tau        : time delay of the gaussian damped time courses
%                . omega      : variation of the gaussian damped time courses
%                . Amp        : Amplitude of the gaussian damped time courses
%                . pQ         : fMRI clusters
% Version 1: Liu Ke, 2019/2/23
%% ===== DEFINE DEFAULT OPTIONS =====
% Def_OPTIONS.WGN         = 1;
% Def_OPTIONS.uniform     = 1;
% Def_OPTIONS.ar          = 0;
% Def_OPTIONS.GridLoc     = [];
% Def_OPTIONS.MixMatrix = eye(4);%eye(numel(tau));
% % Copy default options to OPTIONS structure (do not replace defined values)
% OPTIONS = struct_copy_fields(OPTIONS, Def_OPTIONS, 0);

A = OPTIONS.Amp;
SNR = OPTIONS.SNR;
MixMatrix = OPTIONS.MixMatrix;
pQ = OPTIONS.pQ;
nSource = size(LFM,2);
%% Active Vox    
ActiveVoxSeed = num2cell(pQ);
ActiveVox = [];
Cortex.Faces = Cortex.face;
Cortex.Vertices = Cortex.vert;
[~, VertArea] = tess_area(Cortex);
% Cortex.VertConn = tess_vertconn(Cortex.vert, Cortex.face);
for k = 1:numel(ActiveVoxSeed)
    ActiveVoxSeed{k} = find(pQ{k} ~= 0);
    ActiveVox = union(ActiveVoxSeed{k},ActiveVox);
end
Area = sum(VertArea(ActiveVox));
%% ------------------ Simulation data ---------------------%
StimTime = find(abs(Time) == min(abs(Time)));
s_real = zeros(nSource,numel(Time));
Activetime = StimTime+1:numel(Time);
% -----------Gaussian Damped sinusoidal time courses------------------%
    Basis = OPTIONS.TBFs;
    AA = MixMatrix*A;
    % % ========================Uniform Sources ==============================%
        for k = 1:numel(ActiveVoxSeed)
            s_real(ActiveVoxSeed{k},:) = repmat(AA(k,:)*Basis,numel(ActiveVoxSeed{k}),1);
        end

    % %==================== White Gaussian Noise ============================= %
B = awgn(LFM*s_real,SNR,'measured');
% =======================================================================%
Result.B = B;
Result.real = s_real;
Result.ActiveVox = ActiveVox;
Result.ActiveVoxSeed = ActiveVoxSeed;
Result.Area = Area;
    
