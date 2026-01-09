################################### setting path ###################################
import sys
sys.path.append('../')
sys.path.append('../../')
################################### import libs ###################################
from  pytorchyolo import  models_split_large
import numpy as np
import time
import torch
import math
import os
from split_framework.split_framework_dynamic import SplitFramework
import tqdm
import numpy as np
import requests, pickle
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from terminaltables import AsciiTable
from pytorchyolo.utils.utils import load_classes, ap_per_class, get_batch_statistics, xywh2xyxy
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from manager.manager_full_large import Manager
import random
from scipy.interpolate import griddata
import pandas as pd


import warnings
warnings.filterwarnings("ignore")
################################### Varialbe init ###################################

split_layer= int(sys.argv[1])


cfg_path = "../configs/yolov3.cfg"
model_path = "../ckpt/stmarc_full.pth"
testdata_path = "../../datasets/St_Marc_dataset/test_30_fps_long_cleaned.txt"
class_name_path = "../../datasets/St_Marc_dataset/coco.names"
log_dir = "../measurements_large/"

bw_measurements = "../5G_bw_trace/5G_bw.csv"
test_case = "test_mtlhq"
service_uri = "http://10.0.1.34:8092/tensor"
reset_uri = "http://10.0.1.34:8092/reset"

measurement_path = log_dir+test_case+"/"
map_output_path = measurement_path+ "map.csv"
time_output_path = measurement_path+ "time.csv"
characteristic_output_path = measurement_path+ "characteristic.csv"
manager_output_path = measurement_path + "manager.csv"

if split_layer==10:
    model_split_layer = 11
    dummy_head_tensor = torch.rand([1, 128, 104, 104])
else:
    print("Not supported split layer... Only support split at 10th layer...")
    sys.exit(0)
################################### Clean Old Logs ###################################
try:
    os.mkdir(log_dir)
except:
    os.system("rm -rf "+measurement_path)
    os.system("mkdir -p "+measurement_path)
        
with open(manager_output_path,"a") as f:
    title= (
        "frame_id,"
        "bandwidth,"
        "drop,"
        "target_latency,"
        "technique,"
        "feasibility,"
        "target_cmp,"
        "target_snr,"
        "est_cmp,"
        "est_snr,"
        "pruning_thresh,"
        "quality,"
        "jpeg_F,"
        "decom_F,"
        "reg_F,"
        "opt_time\n"
    )
    f.write(title)

with open(map_output_path,'a') as f:
    title = ("pruning_thresh,"
            "quality,"
            "technique,"
            "bandwidth,"
            "drop,"
            "frame_id,"
            "feasible,"
            "sensitivity,"
            "map\n")
    f.write(title)

with open(time_output_path,'a') as f:
    title = ("pruning_thresh,"
            "quality,"
            "technique,"
            "bandwidth,"
            "drop,"
            "frame_id,"
            "model_head_time,"
            "model_tail_time,"
            "framework_head_time,"
            "framework_tail_time,"
            "framework_response_time,"
            "compression_time,"
            "decompression_time,"
            "overall_time\n"
            )
    f.write(title)

with open(characteristic_output_path,'a') as f:
    title = ("pruning_thresh,"
            "quality,"
            "technique,"
            "bandwidth,"
            "drop,"
            "frame_id,"
            "sparsity,"
            "compression_ratio,"
            "datasize,"
            "reconstruct_snr,"
            "target_cmp,"
            "target_snr,"
            "consumed_bw\n")
    f.write(title)


################################### Utility functions ###################################
def create_data_loader(data_path):
    dataset = ListDataset(data_path, img_size=416, multiscale=False, transform=DEFAULT_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader


def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")
    return precision, recall, AP, f1, ap_class

def write_manager_data(frame_id, bandwidth, drop, target_latency,manager,opt_time):

    with open(manager_output_path,"a") as f:
        f.write(
            str(frame_id)+","
            +str(bandwidth)+","
            +str(drop)+","
            +str(target_latency)+","
            +str(manager.get_compression_technique())+","
            +str(manager.get_feasibility())+","
            +str(manager.get_target_cmp())+","
            +str(manager.get_target_snr())+","
            +str(manager.get_est_cmp())+","
            +str(manager.get_est_snr())+","
            +str(manager.get_pruning_threshold())+","
            +str(manager.get_compression_quality())+","
            +str(manager.get_result_jpeg_f())+","
            +str(manager.get_result_decom_f())+","
            +str(manager.get_result_reg_f())+","
            +str(opt_time)+"\n"
        )

def write_time_data(sf, thresh,quality,tech,bandwidth, drop,frame_id):
    model_head_time, model_tail_time = sf.get_model_time_measurement()
    fw_head_time,fw_tail_time,fw_response_time = sf.get_framework_time_measurement()
    compression_time, decompression_time = sf.get_compression_time_measurement()
    overall_time = sf.get_overall_time_measurement()

    with open(time_output_path,'a') as f:
        f.write(str(thresh)+","
                +str(quality)+","
                +str(tech)+","
                +str(bandwidth)+","
                +str(drop)+","
                +str(frame_id)+","
                +str(model_head_time)+","
                +str(model_tail_time)+","
                +str(fw_head_time)+","
                +str(fw_tail_time)+","
                +str(fw_response_time)+","
                +str(compression_time)+","
                +str(decompression_time)+","
                +str(overall_time)+"\n"
                )
        
def write_characteristic(sf, manager,bandwidth,drop,frame_id, consumed_bw, cmp_ratio):
    sparsity= sf.get_tensor_characteristics()
    datasize_est, _ = sf.get_data_size()
    reconstruct_snr = sf.get_reconstruct_snr()
    target_cmp, target_snr = manager.get_intermedia_measurements()
    # cmp_ratio = (128*26*26*4)/datasize_est

    with open(characteristic_output_path,'a') as f:
        f.write(str(manager.get_pruning_threshold())+","
                +str(manager.get_compression_quality())+","
                +str(manager.get_compression_technique())+","
                +str(bandwidth)+","
                +str(drop)+","
                +str(frame_id)+","
                +str(sparsity)+","
                +str(cmp_ratio)+","
                +str(datasize_est)+","
                +str(reconstruct_snr)+","
                +str(target_cmp)+","
                +str(target_snr)+","
                +str(consumed_bw)+"\n"
                )
        
def write_map( thresh,quality,tech,bandwidth,drop,frame_id,feasibility,sensitivity,map_value):
    with open(map_output_path,'a') as f:
                f.write(str(thresh)+","
                        +str(quality)+","
                        +str(tech)+","
                        +str(bandwidth)+","
                        +str(drop)+","
                        +str(frame_id)+","
                        +str(feasibility)+","
                        +str(sensitivity)+","
                        +str(map_value)+"\n"
                        )
                

################################### Main function ###################################
if __name__ == "__main__":
    # Load Model
    model = models_split_large.load_model(cfg_path, model_path)
    model.set_split_layer(model_split_layer) # layer <7
    model = model.eval()
    bw_df = pd.read_csv(bw_measurements)
    
    dataloader = create_data_loader(testdata_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    class_names = load_classes(class_name_path)  # List of class names

    reset_required = True
    while reset_required:
        r = requests.post(url=reset_uri)
        result = pickle.loads(r.content)
        if result["reset_status"] == True:
            reset_required = False
        else:
            print("Reset edge reference tensor failed...")
        time.sleep(1)

    
    frame_predicts = []
    sf = SplitFramework(device="cuda", model=model)
    sf.set_reference_tensor(dummy_head_tensor)
    manager = Manager()
    
    previouse_bandwidth = 0
    previouse_snr =0
    previouse_drop = 0
    ################## Init measurement lists ##########################
    frame_index = 0
    drop = 0.3
    for _, imgs, targets in tqdm.tqdm(dataloader, desc="testing"):
        frame_index+=1
        target_latency = 0.45 # in seconds
        # availble bandwith calculation
        available_bandwidth = (griddata(bw_df["time"],bw_df["bandwidth_tx"], frame_index*(1/5)+30, method='nearest')/5-20)*1e6
        
        if frame_index%10==0:
            drop = (1-available_bandwidth/(1e7))*100
            drop = round(drop)/100
            drop = max(0.2,drop)
            drop = min(0.5,drop)

            drop = drop+0.1 # only for no jpeg
      
        # technique = 1

        # interframe similarity calculation

        # check framework update events and run manager 
        manager_begin = time.time_ns()
        if frame_index< manager.get_testing_frame_length()+1:
            print("In initial phase")
            manager.update_requirements(drop,target_latency,available_bandwidth,frame_index)
            fesiable = manager.get_feasibility()
            write_manager_data(frame_index,available_bandwidth,drop,target_latency, manager,(time.time_ns()-manager_begin)/1e6)
            previouse_drop = drop
            previouse_bandwidth = available_bandwidth
        elif available_bandwidth!=previouse_bandwidth:
            print("In bandwidth change update")
            manager.update_requirements(drop,target_latency,available_bandwidth,frame_index)
            fesiable = manager.get_feasibility()
            write_manager_data(frame_index,available_bandwidth,drop,target_latency, manager,(time.time_ns()-manager_begin)/1e6)
            previouse_drop = drop
            previouse_bandwidth = available_bandwidth
        elif (manager.get_target_snr() - previouse_snr)/manager.get_target_snr() > 0.3:
            print("In SNR exceed update")
            manager.update_requirements(drop,target_latency,available_bandwidth,frame_index)
            fesiable = manager.get_feasibility()
            write_manager_data(frame_index,available_bandwidth,drop,target_latency, manager,(time.time_ns()-manager_begin)/1e6)
            previouse_drop =drop
            previouse_bandwidth = available_bandwidth
        elif previouse_drop!=drop:
            print("In target drop change update")
            manager.update_requirements(drop,target_latency,available_bandwidth,frame_index)
            fesiable = manager.get_feasibility()
            write_manager_data(frame_index,available_bandwidth,drop,target_latency, manager,(time.time_ns()-manager_begin)/1e6)
            previouse_drop =drop
            previouse_bandwidth = available_bandwidth


        
        # thresh, quality = manager.get_configuration() 
        # set framework configuration
        sf.set_compression_technique(manager.get_compression_technique()) # set to jpeg
        sf.set_quality(manager.get_compression_quality())
        sf.set_pruning_threshold(manager.get_pruning_threshold())


        # Real measurements
        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        # Extract labels
        labels = targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= 416
        # run inference
        detection = sf.split_framework_client(imgs,service_uri=service_uri)



        data_size,_ = sf.get_data_size()
        # print(data_size)
        cmp= (128*104*104*4)/data_size

        manager.update_sample_points(manager.get_compression_technique(),(manager.get_pruning_threshold(),manager.get_compression_quality()),cmp,sf.get_reconstruct_snr())
        previouse_snr = sf.get_reconstruct_snr()

        if manager.get_transmission_time()  == -1:
            consumed_bw =-1
        else:
            consumed_bw = sf.get_data_size()[0]*8 / manager.get_transmission_time() # in bps
        
        
        write_time_data(sf,manager.get_pruning_threshold(),manager.get_compression_quality(),manager.get_compression_technique(),available_bandwidth,drop,frame_index)
        write_characteristic(sf,manager,available_bandwidth,drop,frame_index,consumed_bw,cmp)
        sample_metrics = get_batch_statistics(detection, targets, iou_threshold=0.1)

        # Concatenate sample statistics
        try:
            true_positives, pred_scores, pred_labels = [
                np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
            metrics_output = ap_per_class(
                true_positives, pred_scores, pred_labels, labels)
    
            sensitivity = np.sum(true_positives) / len(labels)
            precision, recall, AP, f1, ap_class = print_eval_stats(metrics_output, class_names, True)
            ## Save data
            write_map(manager.get_pruning_threshold(),manager.get_compression_quality(),manager.get_compression_technique(),available_bandwidth,drop,frame_index,fesiable,sensitivity,AP.mean())
        except:
            write_map(manager.get_pruning_threshold(),manager.get_compression_quality(),manager.get_compression_technique(),manager.get_compression_technique(),available_bandwidth,drop,frame_index,fesiable,0, 0)

        

        