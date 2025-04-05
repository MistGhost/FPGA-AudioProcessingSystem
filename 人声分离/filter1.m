[y, Fs] = audioread('人声分离测试音频样本.wav');    %[y,Fs] = audioread(filename) y为音频数据，Fs为采样率
t = 0:1/Fs:(size(y,1)-1)/Fs;           
f = Fs*(0:4095)/4096;
y1 = fft(y, 4096);
fz = abs(y1(1:4096));     %取y1前2048个构成一维数组

pks = findpeaks(fz);       
[b, c] = sort(pks, 'descend'); %对fz值进行排列 
wz1 = find(fz == b(1));
wz2 = find(fz == b(2));
wz3 = find(fz == b(3));
F1 = wz1/4096*Fs; %找出幅度最大时对应的频率的值，滤除该值
F2 = wz2/4096*Fs;
F3 = wz3/4096*Fs;
fil_1 = filter(Hd, y); %Hbs滤波函数 y为滤波输入

% 放大音频
amplificationFactor = 4; % 可以根据需要调整
amplifiedAudio = fil_1 * amplificationFactor;

% 防止溢出，确保音频值在[-1, 1]之间
amplifiedAudio = min(max(amplifiedAudio, -1), 1);

% 绘制频谱图
figure;

% 原始音频的频谱
subplot(2,1,1);
plot(f, fz);
title('原始音频的频谱');
xlabel('Hz');
ylabel('幅值');
axis([0, 5000, 0, 10]);

% 滤波且放大后的频谱
y3 = fft(amplifiedAudio, 4096);
fz_3 = abs(y3(1:4096));

subplot(2,1,2);
plot(f, fz_3);
title('滤波且放大后的音频频谱');
xlabel('Hz');
ylabel('幅值');
axis([0, 5000, 0, 10]);

% 保存放大的音频
audiowrite('背景_放大.wav', amplifiedAudio, Fs);

disp('Audio amplification completed and saved to 背景_放大.wav');
