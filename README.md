# Fine-tuning Qwen2.5-VL-3B


### News/Updates
* 2025/02/08
  * First version of the fine-tuning code is released.

### Introduction
In the past five months since Qwen2-VL’s release, numerous developers have built new models on the Qwen2-VL vision-language models, providing us with valuable feedback. During this period, qwen team focused on building more useful vision-language models. Today, qwen team are excited to introduce the latest addition to the Qwen family: Qwen2.5-VL.

<img src="readme_imgs/1.jpeg" width="100%" height="40%">   
    
I personally prefer simple and transparent code, so I wrote a fine-tuning code script for Qwen2.5-VL, hoping to help anyone who like to write their own training loops.  

I have a WeChat subscription account "Backpropagation", where I occasionally write some technical articles, including this one ( https://mp.weixin.qq.com/s/mN9Pxpd2Wciw1-IAoFc08A ), welcome to follow.
<img src="readme_imgs/3.jpg" width="20%" height="20%"> 

### Quick Start for Fine-tuning or continue pre-train Qwen2.5-VL 2B Model
---

```bash
%git clone https://github.com/zhangfaen/finetune-Qwen2.5-VL
%cd finetune-Qwen2.5-VL
%conda create --name qwen-vl-2.5 python=3.102
%conda activate qwen-vl-2.5
%pip install -r requirements.txt
```
you can run the following command to begin:
```bash
./finetune_distributed.sh # Note that the CUDA_VISIBLE_DEVICES variable in this file should be set to the appropriate value
```   

### Test the Fine-tuned Model

---
```bash
%python test_on_trained_model_by_us.py # Test our fine-tuned or retrained Qwen2.5-VL 3B model
```   
Note: The test_on_trained_model_by_us.py file defines model_dir. If you have fine-tuned multiple models, you can modify this file to specify the path of your fine-tuned model.



The above test_on_trained_model_by_us.py both describe the two pictures under test_data/.


Overall, the fine-tuned model seems to have not been greatly affected in performance. The following picture is a log file during the fine-tuning process.
<img src="readme_imgs/4.jpg" width="100%" height="70%">   
<img src="readme_imgs/2.jpg" width="100%" height="70%">   
It can be seen that the training loss is decreasing, indicating that the model has converged during the training process.

