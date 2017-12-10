clc, clear, close all

nH = 512;
nV = 384;
% nH = 1216
% nV = 376
%% flownet_S
% conv1
nH1_in = nH;
nV1_in = nV;
p = 3;
s = 2;
f = 7;
c2 = 64;
nV1_out = getOutputSize(nV1_in, p, s, f);
nH2_out = getOutputSize(nH1_in, p, s, f);
fprintf('conv1  , dim: (%d, %d, %d)\n', nV1_out, nH2_out, c2)

% conv2
p = 3;
s = 2;
f = 7;
c3 = 128;
nV2_out = getOutputSize(nV1_out, p, s, f);
nH2_out = getOutputSize(nH2_out, p, s, f);
fprintf('conv2  , dim: (%d, %d, %d)\n', nV2_out, nH2_out, c3)

% conv3
p = 2;
s = 2;
f = 5;
c3_out = 256;
nV3_out = getOutputSize(nV2_out, p, s, f);
nH3_out = getOutputSize(nH2_out, p, s, f);
fprintf('conv3  , dim: (%d, %d, %d)\n', nV3_out, nH3_out, c3_out)

% conv3_1
p = 1;
s = 1;
f = 3;
c3_1_out = 256;
nV3_1_out = getOutputSize(nV3_out, p, s, f);
nH3_1_out = getOutputSize(nH3_out, p, s, f);
fprintf('conv3_1, dim: (%d, %d, %d)\n', nV3_1_out, nH3_1_out, c3_1_out)

% conv4
p = 1;
s = 2;
f = 3;
c4_out = 512;
nV4_out = getOutputSize(nV3_1_out, p, s, f);
nH4_out = getOutputSize(nH3_1_out, p, s, f);
fprintf('conv4  , dim: (%d, %d, %d)\n', nV4_out, nH4_out, c4_out)

% conv4_1
p = 1;
s = 1;
f = 3;
c4_1_out = 512;
nV4_1_out = getOutputSize(nV4_out, p, s, f);
nH4_1_out = getOutputSize(nH4_out, p, s, f);
fprintf('conv4_1, dim: (%d, %d, %d)\n', nV4_1_out, nH4_1_out, c4_1_out)

% conv5
p = 1;
s = 2;
f = 3;
c5_out = 512;
nV5_out = getOutputSize(nV4_1_out, p, s, f);
nH5_out = getOutputSize(nH4_1_out, p, s, f);
fprintf('conv5  , dim: (%d, %d, %d)\n', nV5_out, nH5_out, c5_out)

% conv5_1
p = 1;
s = 1;
f = 3;
c5_1_out = 512;
nV5_1_out = getOutputSize(nV5_out, p, s, f);
nH5_1_out = getOutputSize(nH5_out, p, s, f);
fprintf('conv5_1, dim: (%d, %d, %d)\n', nV5_1_out, nH5_1_out, c5_1_out)

% conv6
p = 1;
s = 2;
f = 3;
c6_out = 1024;
nV6_out = getOutputSize(nV5_1_out, p, s, f);
nH6_out = getOutputSize(nH5_1_out, p, s, f);
fprintf('conv6  , dim: (%d, %d, %d)\n', nV6_out, nH6_out, c6_out)

% conv6_1
p = 1;
s = 1;
f = 3;
c6_1_out = 1024;
nV6_out_1 = getOutputSize(nV6_out, p, s, f);
nH6_out_1 = getOutputSize(nH6_out, p, s, f);
fprintf('conv6_1, dim: (%d, %d, %d)\n', nV6_out_1, nH6_out_1, c6_1_out)



























