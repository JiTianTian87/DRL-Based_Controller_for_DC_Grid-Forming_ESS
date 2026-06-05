%% 1.打开simulink文件,展开成环境

open_system('Env_cpl_buck')
obsInfo = rlNumericSpec([6 1],...
    'LowerLimit',[-inf -inf  -inf -inf -inf  -inf]',...
    'UpperLimit',[-inf -inf  -inf inf inf inf ]');
obsInfo.Name = 'observations';
obsInfo.Description = 'error, error-delay, error-d, id ';
numObservations = obsInfo.Dimension(1);
% 总共生成 N 个动作（例如 61 个）
N = 31;  
% 在 [-1, 1] 上均匀采样
x = linspace(-1, 1, N);  
% 采用立方变换，使得 x 接近 0 的区域变化更小，远离 0 的区域变化更大
y = sign(x) .* abs(x).^3;  
% 将中心设为参考值 50，扩展幅度设为 10（这样最小值 50-10=40，最大值 50+10=60）
actions = 50 + 10 * y;  

% 用生成的动作数组创建 RL 离散动作空间
actInfo = rlFiniteSetSpec(actions);

numActions = N;

env = rlSimulinkEnv('Env_cpl_buck','Env_cpl_buck/RL Agent',...
    obsInfo,actInfo);
env.ResetFcn = @(in)localResetFcn(in);

Ts = 0.0001;
Tf = 0.5;
rng(0)


%% 2.搭建神经网络
dnn = [
    imageInputLayer([obsInfo.Dimension(1) 1 1],...
    'Normalization', 'none', 'Name', 'State')
    fullyConnectedLayer(64, 'Name', 'CriticStateFC1')
    reluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(64, 'Name', 'CriticStateFC2')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(numActions, 'Name', 'output')];

criticOptions = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1);%,'UseDevice','gpu'

critic = rlQValueRepresentation(dnn,obsInfo,actInfo, ...
    'Observation',{'State'},'Action',{'output'},criticOptions);


%% 3.设置训练参数
agentOptions = rlDQNAgentOptions(...
    'SampleTime',Ts,...
    'UseDoubleDQN',true,...
    'TargetSmoothFactor',1e-3,...'TargetUpdateFrequency',500,...
    'ResetExperienceBufferBeforeTraining',true,...
    'DiscountFactor',0.9,...
    'ExperienceBufferLength',2e5,...
    'MiniBatchSize',256);
opt.EpsilonGreedyExploration.Epsilon = 1;
opt.EpsilonGreedyExploration.EpsilonDecay = 0.001;
opt.EpsilonGreedyExploration.EpsilonMin = 0.1;


agent = rlDQNAgent(critic,agentOptions);
maxepisodes = 200;
maxsteps = ceil(Tf/Ts);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'ScoreAveragingWindowLength',20, ...
    'Verbose', false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','EpisodeReward',...
    'StopTrainingValue',37000,...
    'SaveAgentCriteria','EpisodeReward',...
    'SaveAgentValue',37000);

%% 4.训练&加载模型
doTraining = true;
%  %doTraining = false;
if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
    save('text.mat','agent');
end

if doTraining==false
    % Load pretrained agent for the example.
%     load('282.mat','agent');
end
   

rlSimulationOptions('MaxSteps',maxsteps,'StopOnError','on');
experiences = sim(env,agent,rlSimulationOptions);

%% 5.reset部分,重置Vref
function in = localResetFcn(in)
blk = sprintf('Env_cpl_buck/Desired Current');
Id = 50;
while Id <= 40 || Id >= 80
   Id =  50;
end
in = setBlockParameter(in,blk,'Value',num2str(Id));

end
