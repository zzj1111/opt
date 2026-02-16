export RUBRIC_VLLM_URL="http://127.0.0.1:8001/v1/chat/completions"
export RUBRIC_MODEL_NAME="Qwen/Qwen3-8B"
export RUBRIC_VLLM_TIMEOUT_SEC=60

python /home/zha00175/CudaForge_plus/verl/verl/utils/reward_score/offline_test_rubric_reward.py \
  --data_source CudaForgeImprovement \
  --improvement_txt "/home/zha00175/cuda/CudaForge/run/Level 1/2_Standard_matrix_multiplication_/evaluation/llm_io/round001_repair_prompt.txt" \
  --reference_py "/home/zha00175/cuda/CudaForge/KernelBench/level1/2_Standard_matrix_multiplication_.py" \
  --candidate_py "/home/zha00175/cuda/CudaForge/run/Level 1/2_Standard_matrix_multiplication_/code/kernel_20260111_173155.py" \
  --save_trimmed /home/zha00175/CudaForge_plus/verl/verl/utils/reward_score/trimmed.txt
