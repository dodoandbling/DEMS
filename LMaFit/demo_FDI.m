% clear;

opts.tol = 1e-5;
beta = 1:30;

casename = 'case14';
noise_flag = 5;
att_type = 'single';


nomal_data_fn = ['XXX/',casename,'/z_',num2str(noise_flag),'.mat'];
attack_data_fn = ['XXX/',casename,'/',att_type,'/za_',num2str(noise_flag),'.mat'];

z_ori = load(nomal_data_fn).z;
za = load(attack_data_fn).za;

[att_times, no_mea] = size(za);

A = zeros(att_times, no_mea);
Z = zeros(att_times, no_mea);

window = no_mea;

e = [z_ori;z_ori;z_ori;z_ori];
t = e(1:window-1,:);
T = [];
for i = 1:att_times
%for i = 1:att_times
    za_all = [t;za(i,:)];
%     if mod(i,2) == 0
%         za_all = [t;z_ori(i,:)];
%     else
%         za_all = [t;za(i,:)];
%     end
    [m,n] = size(za_all);
    k = 1;
%     tic;
    [X,Y,S,out] = lmafit_sms_v1(za_all,k,opts,beta);
%     toc;
    L = X*Y;
    A(i,:) = S(m,:);
    Z(i,:) = L(m,:);
%     T = [T,toc];
end
% ws = char(string(window));
% A_fn = ['XXX/data/window/a_',ws,att_type,casename,'.mat'];
% A_fn = ['XXX/data/',casename,'/LRMF/a_new/a_new_',att_type,num2str(noise_flag),'.mat'];
Z_fn = ['/Users/dqy/Desktop/code0312/HDS/data/',casename,'/LRMF/z_new/z_new_',att_type,num2str(noise_flag),'.mat'];
% time_fn = ['..\\HDS\\data\\',casename,'\LRMF\time.txt'];
% save (A_fn,"A");
save (Z_fn,"Z");
% 
% meantime = mean(T);
% title = ["case";"num of mea";"meanCPUtime"];
% mea_str = num2str(no_mea);
% time_str = num2str(meantime);
% record = [string(casename); string(no_mea); string(meantime)];
% Table = table(title,record);
% writetable(Table,time_fn);


% imagesc(S,[-20 20]); 
% colormap("colorcube");

