module load singularity
export LITELLM_MODEL_REGISTRY_PATH="registry.json"
export MSWEA_COST_TRACKING="ignore_errors"
export SINGULARITY_CACHEDIR="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/singularity"
export SINGULARITY_TMPDIR="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/singularity/tmp"
mini-extra swebench \
    --output /scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/test_qwen3_coder_v2_basic_reward_qwen3_5_4b \
    --subset verified \
    --split test \
    --redo-existing \
    --config swebench.yaml \
    --config agent.verifier.enabled=true \
    --config agent.candidate_sampling.num_candidates=5 \
    --config agent.verifier.prompt_dir=/home/zkjzou/SWE-PRM/mini-swe-agent/prompts/verifier \
    --config agent.verifier.prompt_name=basic/reward \
    --config agent.verifier.verifier_type=reward_model \
    --config verifier_model_profile=qwen3_5_instruct \
    --config model.model_class="litellm" \
    --config agent_model_profile="qwen3_coder" \
    --config model.model_name="openai/Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --config model.model_kwargs.api_base="http://localhost:8080/v1" \
    --config agent.verifier.model.model_name="openai/Qwen/Qwen3.5-4B" \
    --config agent.verifier.model.model_kwargs.api_base="http://localhost:8081/v1" \
    --workers 4

mini-extra swebench \
    --output /scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/test_qwen3_coder_v2_basic_mini_reward_qwen3_5_4b \
    --subset verified \
    --split test \
    --redo-existing \
    --config swebench.yaml \
    --config agent.verifier.enabled=true \
    --config agent.candidate_sampling.num_candidates=5 \
    --config agent.verifier.prompt_dir=/home/zkjzou/SWE-PRM/mini-swe-agent/prompts/verifier \
    --config agent.verifier.prompt_name=basic_mini/reward \
    --config agent.verifier.verifier_type=reward_model \
    --config verifier_model_profile=qwen3_5_instruct \
    --config model.model_class="litellm" \
    --config agent_model_profile="qwen3_coder" \
    --config model.model_name="openai/Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --config model.model_kwargs.api_base="http://localhost:8080/v1" \
    --config agent.verifier.model.model_name="openai/Qwen/Qwen3.5-4B" \
    --config agent.verifier.model.model_kwargs.api_base="http://localhost:8081/v1" \
    --workers 4

mini-extra swebench \
    --output /scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/test_qwen3_coder_v2_domain_reward_qwen3_5_4b \
    --subset verified \
    --split test \
    --redo-existing \
    --config swebench.yaml \
    --config agent.verifier.enabled=true \
    --config agent.candidate_sampling.num_candidates=5 \
    --config agent.verifier.prompt_dir=/home/zkjzou/SWE-PRM/mini-swe-agent/prompts/verifier \
    --config agent.verifier.prompt_name=domain/reward \
    --config agent.verifier.verifier_type=reward_model \
    --config verifier_model_profile=qwen3_5_instruct \
    --config model.model_class="litellm" \
    --config agent_model_profile="qwen3_coder" \
    --config model.model_name="openai/Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --config model.model_kwargs.api_base="http://localhost:8080/v1" \
    --config agent.verifier.model.model_name="openai/Qwen/Qwen3.5-4B" \
    --config agent.verifier.model.model_kwargs.api_base="http://localhost:8081/v1" \
    --workers 4

mini-extra swebench \
    --output /scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/test_qwen3_coder_v2_domain_v2_reward_qwen3_5_4b \
    --subset verified \
    --split test \
    --redo-existing \
    --config swebench.yaml \
    --config agent.verifier.enabled=true \
    --config agent.candidate_sampling.num_candidates=5 \
    --config agent.verifier.prompt_dir=/home/zkjzou/SWE-PRM/mini-swe-agent/prompts/verifier \
    --config agent.verifier.prompt_name=domain_v2/reward \
    --config agent.verifier.verifier_type=reward_model \
    --config verifier_model_profile=qwen3_5_instruct \
    --config model.model_class="litellm" \
    --config agent_model_profile="qwen3_coder" \
    --config model.model_name="openai/Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --config model.model_kwargs.api_base="http://localhost:8080/v1" \
    --config agent.verifier.model.model_name="openai/Qwen/Qwen3.5-4B" \
    --config agent.verifier.model.model_kwargs.api_base="http://localhost:8081/v1" \
    --workers 4

mini-extra swebench \
    --output /scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/test_qwen3_coder_v2_world_reward_qwen3_5_4b \
    --subset verified \
    --split test \
    --redo-existing \
    --config swebench.yaml \
    --config agent.verifier.enabled=true \
    --config agent.candidate_sampling.num_candidates=5 \
    --config agent.verifier.prompt_dir=/home/zkjzou/SWE-PRM/mini-swe-agent/prompts/verifier \
    --config agent.verifier.prompt_name=world/reward \
    --config agent.verifier.verifier_type=reward_model \
    --config verifier_model_profile=qwen3_5_instruct \
    --config model.model_class="litellm" \
    --config agent_model_profile="qwen3_coder" \
    --config model.model_name="openai/Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --config model.model_kwargs.api_base="http://localhost:8080/v1" \
    --config agent.verifier.model.model_name="openai/Qwen/Qwen3.5-4B" \
    --config agent.verifier.model.model_kwargs.api_base="http://localhost:8081/v1" \
    --workers 4
