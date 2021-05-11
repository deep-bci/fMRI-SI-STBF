function [S,par] = STBFSI(B,L,SBFs,StimTime,cls,varargin)
tic
%% Discription : Estimated the Brain sources based on Spataio-Temporal Basis functions
% B(t) =  L*SBFs * W *Phi(t)+ epsilon;  
% L \in R^{db x ds}, db is #Sensors, ds is #dipoles on the cortical surface.
% S = SBFs * W *Phi; 
% SBFs: Spatial Basis functions; SBFs = [M1, ..., M_nalpha], Mi \in R^{ds
% x Ni}, Ni denotes the number of dipoles in the ith patch, there are
% totally nalpha patches.
% Phi \in R^{K x T}: Temporal Basis functions
% Both W and Phi are unknown and learned from the measurements B.

% prior: 
% p(W_k) = N(0, alpha^{-1});
% p(Phi_t) = N(0, C^{-1});

% Input: 
%      B(d_b x T):             M/EEG Measurement
%      L(d_b x d_s):           Leadfiled Matrix
%      Phi(K x T):             Initial Temporal Basis, may be the vector from SVD of B
%      SBFs(d_s x n):          Spatial Basis functions, may be from DDP of
%      EEG recodings or/and clusters from fMRI activation
% Output:
%     S:                       Estimated Sources

% Author : Liu Ke
% Date: 2016/1/16 
% Reference: [1] MEG source localization of spatially extended generators of epileptic activity: 
% comparing entropic and hierarchical bayesian approaches;
%  [2] Probabilistic algorithms for MEG/EEG source reconstruction using
%  temporal basis functions learned from data.
%% Initial of Algorithm
%  Demension of the Inverse Problem
F = L*SBFs;
[nSensor,nSource] = size(F);    % # Sensors and # Sources
nSnap = size(B,2);              % # Snapshots
MAX_ITER = 300;
epsilon = 1e-6;



FreeEnergyCompute = 1; % Whether compute the freeEnergy
ifplot = 0;            % whether plot the freeenergy

beta = 1; % beta: the inverse of mearement noise variance
C_noise = eye(nSensor)/beta;

Cost_old = 0;

cls = ones(nSource,1);

update = 'Convex'; % update rule for alpha; either 'Convex' or 'EM';

prune = [1e-6, 1e-1]; % prune of SBF and TBFs
% get input argument values
if(mod(length(varargin),2)==1)
    error('Optional parameters should always go by pairs\n');
else
    for i=1:2:(length(varargin)-1)
        switch lower(varargin{i})
            case 'epsilon'   
                epsilon = varargin{i+1}; 
            case 'max_iter'
                MAX_ITER = varargin{i+1};
            case 'freeenergycompute'
                FreeEnergyCompute = varargin{i+1};
            case 'update'
                update = varargin{i+1};
            case 'ifplot'
                ifplot = varargin{i+1};
            case 'cls'
                cls = varargin{i+1};
            case 'prune'
                prune = varargin{i+1};
            case 'kinitial'
                K = varargin{i+1};
            otherwise
                error(['Unrecognized parameter: ''' varargin{i} '''']);
        end
    end
end   


% K = 5;               % Initial number of TBFs
[~,~,D] = svd(B);
Phi = D(:,1:K)';
theta = zeros(nSource,K);
c = ones(K,1);

% Sigma_C = zeros(K);
Sigma_C = 1e-6*eye(K);

clusters = cell(numel(cls),1);

for i = 1:numel(cls)
    if i == 1
        clusters{i} = 1:sum(cls(1:i));
    else
        clusters{i} = sum(cls(1:i-1))+1 : sum(cls(1:i));
    end
end
FF = F;

ncls = numel(clusters);
keeplistTBFs = (1:K)';
keeplistSBFs = (1:ncls)';

alpha = 1e0*(ones(ncls,1) + 1e-3*randn(ncls,1))*trace(F*F')*(trace(Phi'*Phi))/(nSnap*trace(C_noise)); % variance of TBFs coeffients
%% Iteration
for iter = 1:MAX_ITER
    %% Temporal Basis Check
if ~isempty(Phi)
    index1 = find(abs(1./c) > max(abs(1./c))* prune(2));  % Select the TBF hyperparameters
    index2 = find(abs(1./alpha) > max(abs(1./alpha))* prune(1)); % Selcet the SBF hyperparameters

    c = c(index1); alpha = alpha(index2); 
    keeplistTBFs = keeplistTBFs(index1);    
    keeplistSBFs = keeplistSBFs(index2);
    clusters = clusters(index2);
    Phi = Phi(index1,:); K = numel(index1);
    
    Ncls = zeros(numel(clusters),1);
    remaindip = []; gamma = [];
    for i = 1:numel(clusters)
        Ncls(i) = numel(clusters{i});
        remaindip = [remaindip,clusters{i}];
        gamma = [gamma;alpha(i)*ones( numel(clusters{i}) ,1)];
    end
    theta = theta(remaindip,index1);
    F = FF(:,remaindip);
end
    %% W Update
    Diag_C = zeros(K,1);
    Diag_W = zeros(numel(clusters),1);
    
    FAFT = F.*repmat(1./gamma',nSensor,1)*F';
    for k = 1:K
        x = Phi(k,:)*Phi(k,:)' + nSnap*Sigma_C(k,k);
        Sig_B =  FAFT+ C_noise/x;
        %         Sigma_theta = diag(1./alpha) - repmat(1./alpha,1,nSensor).*F'/Sig_B*(F.*repmat(1./alpha',nSensor,1));
        residual = (B)*Phi(k,:)'- F*sum((theta(:,setdiff((1:K),k)).*repmat(Phi(k,:)*Phi(setdiff((1:K),k),:)' + nSnap*Sigma_C(k,setdiff((1:K),k)),size(F,2),1)),2);
        theta(:,k) = repmat(1./gamma,1,nSensor).*F'/Sig_B*residual/x;%Sigma_theta *F'/C_noise * residual;
        

        Diag_C(k) = trace( FAFT/C_noise - FAFT/Sig_B*FAFT/C_noise);
        if strcmpi('EM', update)
            FTBF = zeros(numel(clusters),1);
            for i = 1:numel(clusters)
                FTBF(i) = Ncls(i)/alpha(i) - trace( FF(:,clusters{i})*FF(:,clusters{i})'/Sig_B ) /alpha(i)^2;
            end
            Diag_W = Diag_W + FTBF;
        end
    end
    clear x
  %% TBFs Update
        Sigma_C = inv(theta'*F'/C_noise*F*theta  + diag(Diag_C + c));
        Phi = Sigma_C*theta'*F'/C_noise*(B);
        clear temp
    % C Update
        c = 1./(diag(Sigma_C) + diag(Phi*Phi')/nSnap);
    
    %% alpha Update
    mu = zeros(numel(alpha),1);
for i = 1:numel(clusters)
    if i == 1
        col = 1:sum(Ncls(1:i));
    else
        col = sum(Ncls(1:i-1))+1 : sum(Ncls(1:i));
    end
    
    if strcmpi('EM', update)
        alpha(i) = K*Ncls(i)/( trace(theta(col,:)*theta(col,:)') + Diag_W(i) );
    end
    
    if strcmpi('Convex', update)
        for k = 1:K
            x = Phi(k,:)*Phi(k,:)' + nSnap*Sigma_C(k,k);
            mu(i) = mu(i) + trace( F(:,col)'/(FAFT + eye(size(F,1))/x)*F(:,col) );
        end
        alpha(i) = 1/sqrt( (trace(theta(col,:)*theta(col,:)') )/ mu(i) );
    end
end

    %% Recover theta
temp = zeros(nSource,K);
temp(remaindip,:) = theta;
theta = temp;
    %% 
   %% Free Energy Compute
   if FreeEnergyCompute
       gamma = [];
       for i = 1:numel(clusters)
           gamma = [gamma;alpha(i)*ones( numel(clusters{i}) ,1)];
       end
  %% Free Energy for EM update
  if strcmpi('EM', update)
       Cost(iter) = -.5*trace(B'/C_noise*B) + .5*trace(Phi'/Sigma_C*Phi) + .5*nSnap*(sum(log(c)) + log(det(Sigma_C))) ;
       FAFT = F.*repmat(1./gamma',nSensor,1)*F';
       for k = 1:K
           x = Phi(k,:)*Phi(k,:)' + nSnap*Sigma_C(k,k);
           Cost(iter) = Cost(iter) -.5*(nSensor*log(x) - log(det(C_noise)) + log(det(FAFT + C_noise./x)));
       end
  end
  %% Free Energy for Convex update
  if strcmpi('Convex', update)
          Cost(iter) = -.5*trace(B'/C_noise*B) + trace(Phi'/Sigma_C*Phi) + .5*nSnap*(sum(log(c)) + log(det(Sigma_C)))...
              - .5*trace(theta(remaindip,:)'.*repmat(gamma',K,1)*theta(remaindip,:));
      FAFT = F.*repmat(1./gamma',nSensor,1)*F';
      for k = 1:K
          x = Phi(k,:)*Phi(k,:)' + nSnap*Sigma_C(k,k);
          Cost(iter) = Cost(iter) -.5*(nSensor*log(x) - log(det(C_noise)) + log(det(FAFT + C_noise./x)));
      end
      temp = theta(remaindip,:)'*F'/C_noise*F*theta(remaindip,:);
      Cost(iter) = Cost(iter) - .5*trace(Phi'*temp*Phi) -.5*nSnap*trace(temp*Sigma_C) - .5*nSnap*nSensor*log(2*pi) + .5*nSnap*nSensor*log(beta);
  end
%% Check stop conditon
       MSE = (Cost(iter) - Cost_old)/Cost(iter);
       Cost_old = Cost(iter);
       if ifplot
           figure(1)
           plot(1:iter,Cost(1:iter));
       end
   end
   if abs(MSE) < epsilon
       break;
   end
    fprintf('iter = %g, MSE = %g,  #TBFs = %g, #SBFs = %g, #remaindipols = %g\n', iter,MSE,K,numel(clusters),size(F,2)) 
%     1./c
end

%par.theta = theta;
par.W =  SBFs*theta;
par.Phi = Phi;
par.c = c;
par.alpha = alpha;
par.keeplistTBFs = keeplistTBFs;
par.keeplistSBFs = keeplistSBFs;
par.Cost = Cost;
par.runtime = toc;
S = par.W*Phi;
toc