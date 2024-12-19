clear 
clc
close all

% prepare data for python

% %% MEG
% load('cleanmeg.mat')
% meg = cleanmeg.trial{1}(1:273, :)'; % ? 274-275 weird values
% 
% save('meg.mat', "meg")

%% pupil

load('pupil.mat')
noOut_pupil = filloutliers(pupil', 'pchip', 'movmedian', 1000, 'ThresholdFactor', 1);
smoothPupil = smoothdata(noOut_pupil, 'gaussian', 100);
figure(); hold on
plot(pupil')
plot(smoothPupil)  

save('smoothPupil.mat', "smoothPupil")

%% respiration

load('resp.mat')
figure()
plot(x)


%% check out phase space

foo = 1;

dat_merged = [x, smoothPupil];

[~,eLag,eDim] = phaseSpaceReconstruction(dat_merged)



%% merge them

ZrespPupil = real(x)+1i*smoothPupil;

instVectLength = abs(ZrespPupil);
figure(); plot(instVectLength)

fsample = 300; nsecstot = length(ZrespPupil)/fsample;
xtime = 0:1/fsample:nsecstot-1/fsample;

% see angle
theta = angle(ZrespPupil);
figure(); plot(xtime, theta)
xlabel('time')
ylabel('angle(Z)')

title('angle of pupil and respiration in the complex plane')

%% visualize 3 seconds of recording 3d
nsecs = 100; 
t = 0:1/fsample:nsecs-1/fsample;

% select part of the signal
subsig = ZrespPupil(1:length(t));


figure()
plot3(t, real(subsig), imag(subsig))
xlabel('time')
ylabel('resp')
zlabel('pupil')

%% 

figure()
plot(ZrespPupil)

%%

derXresp = diff(x);

figure; 
plot(derXresp, smoothPupil(2:end))


derPupil = diff(smoothPupil);
figure;
plot(angle(derPupil+1i*x(2:end)))


figure;
plot(smoothPupil, x)



