clear;

opts.tol = 1e-5;
beta = 1:15;

casename = 'case14';
noise_flag = 5;
att_type = 'random';


nomal_data_fn = ['XXX//data/',casename,'/z_',num2str(noise_flag),'.mat'];
attacked_data_fn = ['XXX/data/',casename,'/',att_type,'/za_',num2str(noise_flag),'.mat'];
attack_data_fn = ['XXX/data/',casename,'/',att_type,'/a_',num2str(noise_flag),'.mat'];

z_ori = load(nomal_data_fn).z;
za = load(attacked_data_fn).za;
a = load(attack_data_fn).a;

[att_times, no_mea] = size(za);

A = zeros(att_times, no_mea);
Z = zeros(att_times, no_mea);
e = [z_ori;z_ori;z_ori;z_ori];

W = [fix(no_mea/8), fix(no_mea/4), fix(no_mea/2), no_mea];
W_op = zeros(1,att_times);
win = 'adw';
% window = W(1);
% t = e(1:window-1,:);
for i = 1:att_times
    if i == 1
        window = no_mea;
    else
        window = optimal_window(W,za(i,:),a(i,:),e);
    end
    W_op(i) = window;
    t = e(1:window-1,:);
    za_all = [t;za(i,:)];
    [m,n] = size(za_all);
    k = 3;
    [X,Y,S,out] = lmafit_sms_v1(za_all,k,opts,beta);

    L = X*Y;
    A(i,:) = S(m,:);
    Z(i,:) = L(m,:);
end
w = num2str(win);
% w = 'adw';
A_fn = ['XXX/data/',casename,'/LRMF/a_new/a1_new_', w ,'_',att_type,num2str(noise_flag),'.mat'];
save (A_fn,"A");







