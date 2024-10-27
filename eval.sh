# env_name=MsPacman
# python -u eval.py \
#     -env_name "ALE/${env_name}-v5" \
#     -run_name "${env_name}-life_done-wm_2L512D8H-100k-seed1"\
#     -config_path "config_files/STORM.yaml" 


env_name=MineDojo/Combat_Spider

MINEDOJO_HEADLESS=1 python -u eval.py \
    -env_name "${env_name}" \
    -run_name "${env_name}-life_done-wm_2L512D8H-100k-seed1"\
    -config_path "config_files/STORM.yaml" \