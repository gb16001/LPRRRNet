# LPR<sup>3</sup>Net

Licence Plate Recognition with Residual Recurrent Neural Network
## usage
### dependecy

- please install torch-gpu python env first
- `pip install -r requirements.txt`
### dataset prepare
download [CBLPRD-330k dataset](https://github.com/SunlifeV/CBLPRD-330k?tab=readme-ov-file), and put it in `data/CBLPRD-330k_v1`folder
### train
run `train_net.py` script, the config file of this script is `args.yaml`.  
please make sure the CBLtrain and CBLval keys point to the right path  
by reconfig model_name key, you can try other models built in `model/LPRRRNet.py`  
**caution**: `lpr_class_predict: false lpr_CTC_predict: True`should set to the right value accroding to the model.
### evaluate
run `eval_net.py` script with config file `args_eval.yaml`.
## result
the best model achived 98.6% acc on CBLPRD with a nano size.

| **License Plate Classes** | **Length err(%)** | **Char err(%)** | **Total err(%)** |
| --- | --- | --- | --- |
| Black License Plates | 0.75  | 0.37  | 1.12  |
| Single-layer Yellow Plates | 0.47  | 1.33  | 1.79  |
| Double-layer Yellow Plates | 0.99  | 4.57  | 5.56  |
| Standard Blue Plates | 0.33  | 0.31  | 0.64  |
| Green Plates for Tractors | 0.27  | 2.25  | 2.52  |
| New Energy Large Vehicle  | 0.32  | 0.61  | 0.93  |
| New Energy Small Vehicle  | 0 | 0.42  | 0.42  |

## reference
> [LPRNet](https://github.com/sirius-ai/LPRNet_Pytorch)  
> [CBLPRD dataset](https://github.com/SunlifeV/CBLPRD-330k?tab=readme-ov-file)

