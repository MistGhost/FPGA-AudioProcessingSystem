clear;clc;close all;
[x,fs] = audioread('去噪测试音频样本2.wav');%读取音乐信号

%输出频率
fprintf('fs1: %i \n',fs )
%音乐语音信号分声道处理
x_l=x(:,1);
n1=length(x_l);
t1=(0:(n1-1))/fs;%length取数列长度即元素个数
is_shift=1; % 预留接口，下面用来判断频谱是否使用fftshift
%=======================原音频时域/频域图=================================%
figure('NumberTitle', 'off', 'Name', '原音频时域/频域波形图');
%画音乐信号时域图
subplot(2,1,1);plot(t1,x_l);
axis([0,5,-0.5,0.5]);xlabel('时间t');ylabel('幅度');title('音乐时域波形');
%画音乐信号频域图
x_l_fft=fft(x_l,n1);
if is_shift==1 
x_l_fft=fftshift(x_l_fft);
end
f1=0:fs/n1:fs*(n1-1)/n1;
subplot(2,1,2);plot(f1,abs(x_l_fft));
axis([0,fs,0,200]);xlabel('频率f');ylabel('幅度');title('音乐信号频谱');
%sound(x,fs);pause(5);%解调并用巴特沃斯滤波器滤波后的声音 
%=======================经过带通滤波器=================================%

% 对信号进行滤波
x_l_filtered=bandpass(x_l,[0.15 0.4],'Steepness',0.9,'StopbandAttenuation',80);

%=======================带通滤波后的时域/频域图=============================%
figure('NumberTitle', 'off', 'Name', '带通滤波后的时域/频域波形图');
% 画带通滤波后的时域图
subplot(2,1,1);
plot(t1, x_l_filtered);
axis([0, 5, -0.5, 0.5]);
xlabel('时间t');
ylabel('幅度');
title('带通滤波后的音乐时域波形');
% 画带通滤波后的频域图
x_l_filtered_fft = fft(x_l_filtered, n1);
if is_shift == 1 
x_l_filtered_fft = fftshift(x_l_filtered_fft);
end
subplot(2,1,2);
plot(f1, abs(x_l_filtered_fft));
axis([0, fs, 0, 200]);
xlabel('频率f');
ylabel('幅度');
title('带通滤波后的音乐信号频谱');


filename = 'noise_audio2.wav'; % 新音频文件的文件名
audiowrite(filename, x_l_filtered, fs); % 将滤波后的音频信号保存为新的音频文件

sound(x_l_filtered,fs);pause(5);%解调并用巴特沃斯滤波器滤波后的声音 