clearvars
close all
clc

sourcedir = '/remotedata/AgGross/TBraiC/AlphaSpade/hctsa';
d=dir([sourcedir, '/*trials*']);

target_dir = '/home/balestrieri/TBraiC/MV-eye/STRG_decoding_accuracy/alphaSPADE/HCTSA_prestim_linearclass_extended';
if ~isfolder(target_dir)
    mkdir(target_dir)
end

load(fullfile('/remotedata/AgGross/TBraiC/AlphaSpade', 'OP.mat'))
feat_codes = OP.CodeString;

nthreads = 3;
thisObj = parpool(nthreads);

parfor k0=1:length(d)

    % seed for reproducibility
    rng(42)

    subjID = d(k0).name(1:4);
    tmp_tr = load(fullfile(d(k0).folder, d(k0).name))

    fi=find(tmp_tr.mask_stim_present);
    nt=length(fi);
    feat=1;
    dat=zeros(nt,273,7525,'single');
    for k=1:nt
        tmp_res = load(fullfile(d(k0).folder, [subjID, '_features_' num2str(fi(k)) '.mat']))
        dat(k,:,:)=single(tmp_res.res);
    end

    %replace NaNs
    Y=double(tmp_tr.seen_unseen);
    ncv=10;
    cv=cvpartition(Y,'KFold',ncv);
    bacc = nan(7525, 3);
    for k=1:7525
        if (mod(k,200)==0),disp([k0 k]);end
        dat2=squeeze(dat(:,:,k));
        if (length(find(isnan(dat2(:)))) < 2000)

            %dat2=fillmissing2(dat2,'nearest');       ######################################
            dat2=fillmissing(dat2,'nearest',2);

            bacctmp = nan(ncv, 3);
            for k2=1:ncv

                idtrain=training(cv,k2);
                idtest=test(cv,k2);

                try

                    tmp_true = wrap_linear_class(dat2(idtrain,:), dat2(idtest,:), ...
                                                 Y(idtrain), Y(idtest)); 
                    tmp_shffld = wrap_linear_class(dat2(idtrain,:), dat2(idtest,:), ...
                                                   shuffle(Y(idtrain)), shuffle(Y(idtest)));                 
                    
                    bacctmp(k2, 1) = tmp_true-tmp_shffld;
                    bacctmp(k2, 2) = tmp_true;
                    bacctmp(k2, 3) = tmp_shffld;

                end
             
            end

            bacc(k, :) = nanmean(bacctmp, 1);

        else
        
            bacc(k, :) = 0;
        
        end
    end


    % create table and save csv file
    tbl_bacc = [];
    tbl_bacc.feature = feat_codes;
    tbl_bacc.balanced_accuracy = bacc(:, 2);
    tbl_bacc.delta_accuracy = bacc(:, 1);
    tbl_bacc.random_accuracy = bacc(:, 3);
    tbl_bacc = struct2table(tbl_bacc);
    
    writetable(tbl_bacc, fullfile(target_dir, [subjID, '.csv']))
    disp(subjID)

end

delete(thisObj)

