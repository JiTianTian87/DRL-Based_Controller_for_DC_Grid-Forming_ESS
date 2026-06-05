clear;
clc;
%% 1.打开simulink文件,展开成环境
open_system('DUs_Exp') 
numObs = 9;  
numAct = 2;   

obsInfo = rlNumericSpec([numObs 1]);

actInfo = rlNumericSpec([numAct 1]);
actInfo.LowerLimit = -0.45;
actInfo.UpperLimit = -0.05;

env = rlSimulinkEnv('DUs_Exp','DUs_Exp/RL Agent',obsInfo,actInfo);

env.ResetFcn = @(in)localResetFcn(in);

Ts = 0.001;%0.001
Tf = 0.3;
rng(0)

% Define the network layers.
cnet = [
    featureInputLayer(numObs,"Normalization","none","Name","observation")
    fullyConnectedLayer(32,"Name","fc1")
    concatenationLayer(1,2,"Name","concat")
    reluLayer("Name","relu1")
    fullyConnectedLayer(16,"Name","fc3")
    reluLayer("Name","relu2")
    fullyConnectedLayer(1,"Name","CriticOutput")];
actionPath = [
    featureInputLayer(numAct,"Normalization","none","Name","action")
    fullyConnectedLayer(32,"Name","fc2")];

% Connect the layers.
criticNetwork = layerGraph(cnet);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = connectLayers(criticNetwork,"fc2","concat/in2");

criticdlnet = dlnetwork(criticNetwork,'Initialize',false);
criticdlnet1 = initialize(criticdlnet);
criticdlnet2 = initialize(criticdlnet);

critic1 = rlQValueFunction(criticdlnet1,obsInfo,actInfo, ...
    "ObservationInputNames","observation");
critic2 = rlQValueFunction(criticdlnet2,obsInfo,actInfo, ...
    "ObservationInputNames","observation");

% Create the actor network layers.
anet = [
    featureInputLayer(numObs,"Normalization","none","Name","observation")
    fullyConnectedLayer(32,"Name","fc1")

    tanhLayer("Name","tanh1")
    fullyConnectedLayer(16,"Name","fc2")

    tanhLayer("Name","tanh2")];

meanPath = [
    fullyConnectedLayer(numAct,"Name","mean")];
stdPath = [
    fullyConnectedLayer(numAct,"Name","stdFC")
    softplusLayer("Name","std")];

% Connect the layers.
actorNetwork = layerGraph(anet);

actorNetwork = addLayers(actorNetwork,meanPath);
actorNetwork = addLayers(actorNetwork,stdPath);
actorNetwork = connectLayers(actorNetwork,"tanh2","mean/in");
actorNetwork = connectLayers(actorNetwork,"tanh2","stdFC/in");
actordlnet = dlnetwork(actorNetwork);
%plot(actorNetwork)
plot(criticNetwork)
actor = rlContinuousGaussianActor(actordlnet, obsInfo, actInfo, ...
    "ObservationInputNames","observation", ...
    "ActionMeanOutputNames","mean", ...
    "ActionStandardDeviationOutputNames","std");

agentOpts = rlSACAgentOptions( ...
    "SampleTime",Ts, ...
    "TargetSmoothFactor",1e-3, ...        
    "ExperienceBufferLength",1e6, ...      
    "MiniBatchSize",256, ...         
    "NumWarmStartSteps",128, ...    
    "DiscountFactor",0.99); %0.96

agentOpts.EntropyWeightOptions.TargetEntropy = -10;
agentOpts.EntropyWeightOptions.LearnRate = 3e-4;
agentOpts.EntropyWeightOptions.Algorithm = "adam";
agentOpts.EntropyWeightOptions.OptimizerParameters.GradientDecayFactor = 0.95;

% "UseExplorationPolicy",false
    
agentOpts.ActorOptimizerOptions.Algorithm = "adam";
agentOpts.ActorOptimizerOptions.LearnRate = 1e-3;
agentOpts.ActorOptimizerOptions.GradientThreshold = 1;

for ct = 1:2
    agentOpts.CriticOptimizerOptions(ct).Algorithm = "adam";
    agentOpts.CriticOptimizerOptions(ct).LearnRate = 1e-3;
    agentOpts.CriticOptimizerOptions(ct).GradientThreshold = 1;


end

agent = rlSACAgent(actor,[critic1,critic2],agentOpts);

maxepisodes = 700 ;
maxsteps = ceil(Tf/Ts);
trainOpts = rlTrainingOptions(...
    "MaxEpisodes", maxepisodes, ...
    "MaxStepsPerEpisode", maxsteps, ...
    "ScoreAveragingWindowLength", 100, ...
    "Plots", "training-progress", ...
    "StopTrainingCriteria", "AverageReward", ...
    "StopTrainingValue", 50000, ...
    "UseParallel", false);

doTraining = true;
%doTraining = false;
if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
    % gamma_action_base_return
    save('sac.mat','agent');   
end  

if doTraining==false
    % Load pretrained agent for the example.
load('sim335sac.mat','agent');
%load('0616sac_335.mat','agent');
end
   
%通过仿真针对模型验证学习的智能体
rlSimulationOptions('MaxSteps',maxsteps,'StopOnError','on');
experiences = sim(env,agent);


%% 6.reset部分,重置
function in = localResetFcn(in)
blk = sprintf('DUs_Exp/Desired Voltage');
V = 150;
while V <= 2.7 || V >= 500
   V =  150;
end
in = setBlockParameter(in,blk,'Value',num2str(V));
end