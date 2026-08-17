The experiment on ScienceQA dataset.

## 1) Installation
You can install the required dependencies using the following command:
```
conda create -n o3 python=3.10
conda activate o3
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
pip install python-dateutil
```

## 2) Training OOD module
To train the OOD module, you can use the following command:
```
bash ./train_ood.sh
```

## 3) Training orthogonal-regularized LoRA
To unlearn with orthogonal-regularized LoRA, you can use the following command:
```
bash ./train_unlearn_lora.sh
```

## 4) Soft-weighted inference
To run the experiments, you can use the following command:
```
bash eval_soft_infer.sh
```
To gather the final results:
```
python read_results_oodlora.py
```

