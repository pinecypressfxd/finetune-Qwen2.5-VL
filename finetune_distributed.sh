
### multi gpu ###
TORCH_DISTRIBUTED_DEBUG=DETAIL ACCELERATE_DEBUG_VERBOSITY="debug" CUDA_VISIBLE_DEVICES="4,5" accelerate launch --main_process_port=0 --mixed_precision=bf16 --dynamo_backend=no --num_machines=1 --num_processes=2 --use_deepspeed finetune_distributed.py 

### single gpu ###
# CUDA_VISIBLE_DEVICES="4" accelerate launch --main_process_port=0 --mixed_precision=bf16 --dynamo_backend=no --num_machines=1 --num_processes=1 --use_deepspeed finetune_distributed.py 
