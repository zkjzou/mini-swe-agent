module load singularity
export LITELLM_MODEL_REGISTRY_PATH="registry.json"
export MSWEA_COST_TRACKING="ignore_errors"
export SINGULARITY_CACHEDIR="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/singularity"
export SINGULARITY_TMPDIR="/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/singularity/tmp"
mini-extra swebench \
    --output /scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/test_qwen3_coder_v2_basic_verify_qwen3_5_35b \
    --subset verified \
    --split test \
    --redo-existing \
    --config swebench.yaml \
    --config agent.verifier.enabled=true \
    --config agent.candidate_sampling.num_candidates=5 \
    --config agent.verifier.prompt_dir=/home/zkjzou/SWE-PRM/mini-swe-agent/prompts/verifier \
    --config agent.verifier.prompt_name=basic/verifier \
    --config agent.verifier.verifier_type=llm \
    --config verifier_model_profile=qwen3_5_instruct \
    --config model.model_class="litellm" \
    --config agent_model_profile="qwen3_coder" \
    --config model.model_name="openai/Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --config model.model_kwargs.api_base="http://localhost:8080/v1" \
    --config agent.verifier.model.model_name="openai/Qwen/Qwen3.5-35B-A3B" \
    --config agent.verifier.model.model_kwargs.api_base="http://localhost:8081/v1" \
    --enable-langfuse \
    --workers 4

mini-extra swebench \
    --output /scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/test_qwen3_coder_v2_basic_mini_verify_qwen3_5_35b \
    --subset verified \
    --split test \
    --redo-existing \
    --config swebench.yaml \
    --config agent.verifier.enabled=true \
    --config agent.candidate_sampling.num_candidates=5 \
    --config agent.verifier.prompt_dir=/home/zkjzou/SWE-PRM/mini-swe-agent/prompts/verifier \
    --config agent.verifier.prompt_name=basic_mini/verifier \
    --config agent.verifier.verifier_type=llm \
    --config verifier_model_profile=qwen3_5_instruct \
    --config model.model_class="litellm" \
    --config agent_model_profile="qwen3_coder" \
    --config model.model_name="openai/Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --config model.model_kwargs.api_base="http://localhost:8080/v1" \
    --config agent.verifier.model.model_name="openai/Qwen/Qwen3.5-35B-A3B" \
    --config agent.verifier.model.model_kwargs.api_base="http://localhost:8081/v1" \
    --enable-langfuse \
    --workers 4

mini-extra swebench \
    --output /scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/test_qwen3_coder_v2_domain_verify_qwen3_5_35b \
    --subset verified \
    --split test \
    --redo-existing \
    --config swebench.yaml \
    --config agent.verifier.enabled=true \
    --config agent.candidate_sampling.num_candidates=5 \
    --config agent.verifier.prompt_dir=/home/zkjzou/SWE-PRM/mini-swe-agent/prompts/verifier \
    --config agent.verifier.prompt_name=domain/verifier \
    --config agent.verifier.verifier_type=llm \
    --config verifier_model_profile=qwen3_5_instruct \
    --config model.model_class="litellm" \
    --config agent_model_profile="qwen3_coder" \
    --config model.model_name="openai/Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --config model.model_kwargs.api_base="http://localhost:8080/v1" \
    --config agent.verifier.model.model_name="openai/Qwen/Qwen3.5-35B-A3B" \
    --config agent.verifier.model.model_kwargs.api_base="http://localhost:8081/v1" \
    --enable-langfuse \
    --workers 4

mini-extra swebench \
    --output /scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/test_qwen3_coder_v2_domain_v2_verify_qwen3_5_35b \
    --subset verified \
    --split test \
    --redo-existing \
    --config swebench.yaml \
    --config agent.verifier.enabled=true \
    --config agent.candidate_sampling.num_candidates=5 \
    --config agent.verifier.prompt_dir=/home/zkjzou/SWE-PRM/mini-swe-agent/prompts/verifier \
    --config agent.verifier.prompt_name=domain_v2/verifier \
    --config agent.verifier.verifier_type=llm \
    --config verifier_model_profile=qwen3_5_instruct \
    --config model.model_class="litellm" \
    --config agent_model_profile="qwen3_coder" \
    --config model.model_name="openai/Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --config model.model_kwargs.api_base="http://localhost:8080/v1" \
    --config agent.verifier.model.model_name="openai/Qwen/Qwen3.5-35B-A3B" \
    --config agent.verifier.model.model_kwargs.api_base="http://localhost:8081/v1" \
    --enable-langfuse \
    --workers 4

mini-extra swebench \
    --output /scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/SWE-PRM/test_qwen3_coder_v2_world_verify_qwen3_5_35b \
    --subset verified \
    --split test \
    --redo-existing \
    --config swebench.yaml \
    --config agent.verifier.enabled=true \
    --config agent.candidate_sampling.num_candidates=5 \
    --config agent.verifier.prompt_dir=/home/zkjzou/SWE-PRM/mini-swe-agent/prompts/verifier \
    --config agent.verifier.prompt_name=world/verifier \
    --config agent.verifier.verifier_type=llm \
    --config verifier_model_profile=qwen3_5_instruct \
    --config model.model_class="litellm" \
    --config agent_model_profile="qwen3_coder" \
    --config model.model_name="openai/Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --config model.model_kwargs.api_base="http://localhost:8080/v1" \
    --config agent.verifier.model.model_name="openai/Qwen/Qwen3.5-35B-A3B" \
    --config agent.verifier.model.model_kwargs.api_base="http://localhost:8081/v1" \
    --enable-langfuse \
    --workers 4
