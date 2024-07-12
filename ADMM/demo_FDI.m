clear;

opts.loss = 'l1'; 
opts.DEBUG = 0;


casename = 'case118';
noise_flag = 5;
att_type = 'single';


nomal_data_fn = ['..\\HDS\\data\\',casename,'\\z_',num2str(noise_flag),'.mat'];
attack_data_fn = ['..\\HDS\\data\\',casename,'\\',att_type,'\\za_',num2str(noise_flag),'.mat'];

z_ori = load(nomal_data_fn).z;
za = load(attack_data_fn).za;

[att_times, no_mea] = size(za);

A = zeros(att_times, no_mea);
Z = zeros(att_times, no_mea);

window = 30;

e = [z_ori;z_ori;z_ori;z_ori];
w = e(1:window-1,:);
T = [];
% for i = 1:2
for i = 1:att_times
    za_all = [w;za(i,:)];
%     if mod(i,2) == 0
%         za_all = [t;z_ori(i,:)];
%     else
%         za_all = [t;za(i,:)];
%     end
    [m,n] = size(za_all);
    lambda = 1/sqrt(max(m,n));
    tic;
    [L,S,obj,err,iter] = rpca(za_all,lambda,opts);
    toc;
    A(i,:) = S(m,:);
    Z(i,:) = L(m,:);
    T = [T,toc];
end

meantime = mean(T);
% tstr = tostring(meantime)
% ws = char(string(window));
% A_fn = ['..\\data\\window\\a_',ws,att_type,casename,'.mat'];
A_fn = ['..\\HDS\\data\\',casename,'\ADMM\a_new\a_new_',att_type,num2str(noise_flag),'.mat'];
Z_fn = ['..\\HDS\\data\\',casename,'\ADMM\z_new\z_new_',att_type,num2str(noise_flag),'.mat'];
time_fn = ['..\\HDS\\data\\',casename,'\ADMM\time.txt'];
% save (A_fn,"A");
% save (Z_fn,"Z");

title = ["case";"num of mea";"meanCPUtime"];
% mea_str = num2str(no_mea);
% time_str = num2str(meantime);
record = [string(casename); string(no_mea); string(meantime)];
Table = table(title,record);
writetable(Table,time_fn);

% imagesc(S,[-20 20]); 
% colormap("colorcube");

