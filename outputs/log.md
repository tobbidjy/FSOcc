# 3070
time:20240709
instruct:
bash tools/dist_test.sh ./projects/configs/surroundocc/surroundocc.py ckpts/r101_dcn_fcos3d_pretrain.pth 1
out:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 6019/6019, 1.2 task/s, elapsed: 5220s, ETA:     0s
{'IoU': 0.06074218531653693, 'barrier': 0.00029989537388889586, 'bicycle': 2.4123577044004084e-05, 'bus': 0.00030676033069241664, 'car': 0.0019403835834713565, 'construction_vehicle': 0.00010114332530852337, 'motorcycle': 3.0290856236011397e-05, 'pedestrian': 0.0003303726580252569, 'traffic_cone': 1.600906484027748e-05, 'trailer': 0.00020138670126770996, 'truck': 0.0010208250462139475, 'driveable_surface': 0.005477772091912518, 'other_flat': 0.0005279183597377043, 'sidewalk': 0.0006426306471250886, 'terrain': 0.004044852470695331, 'manmade': 0.004399807075671969, 'vegetation': 0.015693041776269956, 'mIoU': 0.0021910758086500603}

time:20240715
instruct:
bash tools/dist_test.sh ./projects/configs/surroundocc/surroundocc.py ckpts/surroundocc.pth 1
out:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 6019/6019, 1.2 task/s, elapsed: 5221s, ETA:     0s
{'IoU': 0.31494956201431085, 'barrier': 0.20597193910271347, 'bicycle': 0.1167754349076627, 'bus': 0.2805863830206866, 'car': 0.3086181381852831, 'construction_vehicle': 0.10699426550168059, 'motorcycle': 0.15138508292315483, 'pedestrian': 0.1408993573514182, 'traffic_cone': 0.12063548968367027, 'trailer': 0.14376548743115142, 'truck': 0.22254564100118052, 'driveable_surface': 0.3729063669244027, 'other_flat': 0.23699815150460668, 'sidewalk': 0.24490486451666652, 'terrain': 0.22769017874717104, 'manmade': 0.14887162099815177, 'vegetation': 0.21863273383263657, 'mIoU': 0.2030113209770148}

time:20241126
instruct:
bash tools/dist_test.sh ./projects/configs/surroundocc/surroundocc_r50.py ckpts/r50_24epoch/epoch_22.pth 1
out:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 6019/6019, 1.6 task/s, elapsed: 3776s, ETA:     0s{'IoU': 0.31351771587053895, 'barrier': 0.18983475144717554, 'bicycle': 0.08102118606506516, 'bus': 0.2575993944126352, 'car': 0.28917869548968234, 'construction_vehicle': 0.064031357177854, 'motorcycle': 0.1117032451434894, 'pedestrian': 0.1173351118889932, 'traffic_cone': 0.09333666446850479, 'trailer': 0.11770206237228566, 'truck': 0.20280348588450747, 'driveable_surface': 0.38792129474946907, 'other_flat': 0.21071127306185994, 'sidewalk': 0.24456704516210906, 'terrain': 0.22957718476034772, 'manmade': 0.13172427581108465, 'vegetation': 0.20980819235431553, 'mIoU': 0.18367845126558616}

# 4090D

# r101
time:20241114 
instruct:
bash tools/dist_test.sh ./projects/configs/surroundocc/surroundocc.py ckpts/surroundocc.pth 2
out:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 6019/6019, 2.8 task/s, elapsed: 2129s, ETA:     0s{'IoU': 0.31494972494481516, 'barrier': 0.2059652830697823, 'bicycle': 0.11678585952765903, 'bus': 0.2805873151225813, 'car': 0.3086182380155797, 'construction_vehicle': 0.10700092158263697, 'motorcycle': 0.15140396903735256, 'pedestrian': 0.14089260719263846, 'traffic_cone': 0.12063671012660615, 'trailer': 0.14376358443519677, 'truck': 0.2225447345973787, 'driveable_surface': 0.3729057563934378, 'other_flat': 0.23700007679819607, 'sidewalk': 0.24490477077439907, 'terrain': 0.22769022074252626, 'manmade': 0.14887141348425126, 'vegetation': 0.21863237680469635, 'mIoU': 0.20301273985655743}
time:20241114 
instruct:
bash tools/dist_test.sh ./projects/configs/surroundocc/surroundocc.py ckpts/surroundocc.pth 1
out:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 6019/6019, 1.8 task/s, elapsed: 3423s, ETA:     0s{'IoU': 0.31494972494481516, 'barrier': 0.2059652830697823, 'bicycle': 0.11678585952765903, 'bus': 0.2805873151225813, 'car': 0.3086182380155797, 'construction_vehicle': 0.10700092158263697, 'motorcycle': 0.15140396903735256, 'pedestrian': 0.14089260719263846, 'traffic_cone': 0.12063671012660615, 'trailer': 0.14376358443519677, 'truck': 0.2225447345973787, 'driveable_surface': 0.3729057563934378, 'other_flat': 0.23700007679819607, 'sidewalk': 0.24490477077439907, 'terrain': 0.22769022074252626, 'manmade': 0.14887141348425126, 'vegetation': 0.21863237680469635, 'mIoU': 0.20301273985655743}


# r50
time:20250223 
batchsize=1
instruct:
bash tools/dist_test.sh ./projects/configs/surroundocc/surroundocc_r50.py ckpts/r50_24epoch_18.37/epoch_22.pth 1
out:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 6019/6019, 2.1 task/s, elapsed: 2877s, ETA:     0s{'IoU': 0.3135167254505816, 'barrier': 0.18982846190797453, 'bicycle': 0.08103047091412742, 'bus': 0.2576070548538193, 'car': 0.2891773880032622, 'construction_vehicle': 0.06403978217014561, 'motorcycle': 0.11172034200962164, 'pedestrian': 0.11733249315883192, 'traffic_cone': 0.09332496702862553, 'trailer': 0.11770553799753478, 'truck': 0.20279932621079155, 'driveable_surface': 0.3879207809387598, 'other_flat': 0.21070934979550426, 'sidewalk': 0.24456640579845706, 'terrain': 0.22957684752827373, 'manmade': 0.13172419796181642, 'vegetation': 0.20980607363649437, 'mIoU': 0.1836793424946275}
time:20250301
batchsize=8
instruct:
bash tools/dist_test.sh ./projects/configs/surroundocc/surroundocc_r50.py ckpts/r50_24epoch_18.37/epoch_22.pth 1
out:
[>>>>>>                                            ] 728/6019, 2.2 task/s, elapsed: 332s, ETA:  2416s

# r50_DSC1
time:20250223 
batchsize=1
instruct:
bash tools/dist_test.sh ./projects/configs/surroundocc/surroundocc_r50.py ckpts/r50_DSC1_18.39/epoch_23.pth 1
out:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 6019/6019, 2.1 task/s, elapsed: 2853s, ETA:     0s{'IoU': 0.31200429923128126, 'barrier': 0.18781084458905828, 'bicycle': 0.08046012207320309, 'bus': 0.25695372040641584, 'car': 0.28814438631582073, 'construction_vehicle': 0.07646984840784606, 'motorcycle': 0.11478289083881159, 'pedestrian': 0.11747973410192387, 'traffic_cone': 0.09808897234494723, 'trailer': 0.12599950372487606, 'truck': 0.20102714769535074, 'driveable_surface': 0.3894579299017981, 00946331588593, 'mIoU': 0.18390177574599992}
time:20250301
batchsize=8
instruct:
bash tools/dist_test.sh ./projects/configs/surroundocc/surroundocc_r50.py ckpts/r50_DSC1_18.39/epoch_23.pth 1
out:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 6019/6019, 2.2 task/s, elapsed: 2765s, ETA:     0s{'IoU': 0.30744399771878034, 'barrier': 0.18782793543085538, 'bicycle': 0.08046131013727996, 'bus': 0.2423369946397795, 'car': 0.2866100049420461, 'construction_vehicle': 0.0778397060229772, 'motorcycle': 0.1138338349707536, 'pedestrian': 0.11687096412603773, 'traffic_cone': 0.09761152343967329, 'trailer': 0.11776989575221354, 'truck': 0.19816512256546137, 'driveable_surface': 0.38933085100693166, 'other_flat': 0.20733795273449446, 'sidewalk': 0.2402539087343536, 'terrain': 0.22479608811267116, 'manmade': 0.12212399756439224, 'vegetation': 0.18500020642055243, 'mIoU': 0.18051064353752957}

# r50_DSC2
time:20250303
batchsize=1
instruct:
bash tools/dist_test.sh ./projects/configs/surroundocc/surroundocc_r50.py ckpts/log_train1/epoch_3.pth 1
out:
[>>>>>>>>>>>>>>>>>>>>                              ] 2408/6019, 2.0 task/s, elapsed: 1204s, ETA:  1806s




# r50_DSC_DSD
time:20250228 
instruct:
batchsize=1
bash tools/dist_test.sh ./projects/configs/surroundocc/surroundocc_r50.py ckpts/r50_DSC_DSD_18.27/epoch_22.pth 1
out:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 6019/6019, 2.0 task/s, elapsed: 3010s, ETA:     0s{'IoU': 0.30945317857740845, 'barrier': 0.1866930251230009, 'bicycle': 0.09364334680790377, 'bus': 0.25917337810449165, 'car': 0.2820345416994086, 'construction_vehicle': 0.08413433427809854, 'motorcycle': 0.10338134430727022, 'pedestrian': 0.1212215001895141, 'traffic_cone': 0.10316461088444388, 'trailer': 0.11778699270175534, 'truck': 0.2050496513621822, 'driveable_surface': 0.3862278580150404, 'other_flat': 0.19644177486525835, 'sidewalk': 0.23695804301296666, 'terrain': 0.2187301156172615, 'manmade': 0.12675363793542965, 'vegetation': 0.20198469853363574, 'mIoU': 0.18271117833985384}
time:20250301
instruct:
batchsize=8
bash tools/dist_test.sh ./projects/configs/surroundocc/surroundocc_r50.py ckpts/r50_DSC_DSD_18.27/epoch_22.pth 1
out:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 6019/6019, 2.2 task/s, elapsed: 2687s, ETA:     0s{'IoU': 0.3046795992500247, 'barrier': 0.18646605828689028, 'bicycle': 0.09360481277051978, 'bus': 0.2445375397722227, 'car': 0.28039547283552485, 'construction_vehicle': 0.08599658796477468, 'motorcycle': 0.10236519362564098, 'pedestrian': 0.12051338966011828, 'traffic_cone': 0.10160860790331633, 'trailer': 0.11258833244660886, 'truck': 0.20276140215784863, 'driveable_surface': 0.38600494573334543, 'other_flat': 0.19699059632354401, 'sidewalk': 0.23661847885339177, 'terrain': 0.21816894959102134, 'manmade': 0.12177537045179833, 'vegetation': 0.18120523240593084, 'mIoU': 0.17947506067390606}

# r50_DSC_DSD2
time:20250304
batchsize=1
instruct:
bash tools/dist_test.sh ./projects/configs/surroundocc/surroundocc_r50.py ckpts/log_train1/epoch_2.pth 1
out:
[>>>>>>>>>>                                        ] 1216/6019, 2.3 task/s, elapsed: 526s, ETA:  2077s

# r50_FastOccHead
time:20250309
batchsize=8
instruct:
bash tools/dist_test.sh ./projects/configs/surroundocc/surroundocc_r50.py ckpts/r50_FastOccHead/epoch_23.pth 1
out:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>                 ] 4000/6019, 2.2 task/s, elapsed: 1785s, ETA:   901s

# r50_hunxi
time:20250309
batchsize=8
instruct:
bash tools/dist_test.sh ./projects/configs/surroundocc/surroundocc_r50.py ckpts/r50_hunxi/epoch_29.pth 1
out:
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 6020/6019, 3.6 task/s, elapsed: 1663s, ETA:     0s
time:20250324
batchsize=8
instruct:
bash tools/dist_test.sh ./projects/configs/surroundocc/surroundocc_r50.py ckpts/r50_hunxi/epoch_35.pth 1
out:
IoU: 0.3150, barrier: 0.1905, bicycle: 0.1100, bus: 0.2570, car: 0.2902, construction_vehicle: 0.0906, motorcycle: 0.1210, pedestrian: 0.1208, traffic_cone: 0.1074, trailer: 0.1188, truck: 0.2025, driveable_surface: 0.3887, other_flat: 0.2123, sidewalk: 0.2397, terrain: 0.2217, manmade: 0.1350, vegetation: 0.2066, mIoU: 0.1883

