# env_name=MsPacman
# python -u eval.py \
#     -env_name "ALE/${env_name}-v5" \
#     -run_name "${env_name}-life_done-wm_2L512D8H-100k-seed1"\
#     -config_path "config_files/STORM.yaml" 

# -run_name "${env_name}-life_done-wm_2L512D8H-100k-seed1"\
env_name=HuntCow
# MINEDOJO_HEADLESS=1 python -u eval.py -log "${env_name}-100k-seed1-JEPA_WM" -seed 1 -config "config_files/100K_CombatSpider_JEPA-WM.yaml" 
# MINEDOJO_HEADLESS=1 python -u eval.py -log "${env_name}-100k-seed1-JEPA_WM-test" -seed 1 -config "Experiments/100K_HuntCow_JEPA-WM_test.yaml" 
# MINEDOJO_HEADLESS=1 python -u eval.py -log "${env_name}-100k-seed1-STORM" -seed 1 -config "Experiments/100K_HuntCow_STORM.yaml"
MINEDOJO_HEADLESS=1 python -u eval.py -log "${env_name}-100k-seed1-JEPA_WM" -seed 1 -config "Experiments/100K_HuntCow_JEPA-WM.yaml" 
# MINEDOJO_HEADLESS=1 python -u eval.py -log "${env_name}-100k-seed1-STORM" -seed 1 -config "config_files/100K_CombatSpider_STORM.yaml" 
