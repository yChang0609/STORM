# env_name=MsPacman
# python -u eval.py \
#     -env_name "ALE/${env_name}-v5" \
#     -run_name "${env_name}-life_done-wm_2L512D8H-100k-seed1"\
#     -config_path "config_files/STORM.yaml" 

# -run_name "${env_name}-life_done-wm_2L512D8H-100k-seed1"\
env_name=HuntCow # HuntCow #CombatSpider
# MINEDOJO_HEADLESS=1 python -u eval.py -log "${env_name}-100k-seed1-JEPA_WM" -seed 1 -config "config_files/100K_CombatSpider_JEPA-WM.yaml" 
# MINEDOJO_HEADLESS=1 python -u eval.py -log "${env_name}-100k-seed1-JEPA_WM-test" -seed 1 -config "Experiments/100K_HuntCow_JEPA-WM_test.yaml" 
# MINEDOJO_HEADLESS=1 python -u eval.py -log "${env_name}-100k-seed1-STORM" -seed 1 -config "Experiments/100K_HuntCow_STORM.yaml"
# MINEDOJO_HEADLESS=1 python -u eval.py -log "${env_name}-100k-seed1-JEPA_WM" -seed 1 -config "Experiments/100K_HuntCow_JEPA-WM.yaml" 
# MINEDOJO_HEADLESS=1 python -u eval.py -log "${env_name}-100k-seed1-STORM" -seed 1 -config "config_files/100K_CombatSpider_STORM.yaml" 
# MINEDOJO_HEADLESS=1 python -u eval.py -log "${env_name}-100k-seed1-JEPA_WM" -seed 1 -config "Experiments/100K_HuntCow_JEPA-WM.yaml" 
# MINEDOJO_HEADLESS=1 python -u eval.py -log "${env_name}-100k-seed1-JEPA_WM-symlog" -seed 1 -config "Experiments/100K_CombatSpider_JEPA-WM-symlog.yaml"
# MINEDOJO_HEADLESS=1 python -u eval.py -log "${env_name}-100k-seed1-STORM-default" -seed 1 -config "Experiments/100K_CombatSpider_STORM-default.yaml"
# MINEDOJO_HEADLESS=1 python -u eval.py -log "CombatSpider-100k-seed1-JEPA-WM-default" -seed 1 -config "Experiments/100K_CombatSpider_JEPA-WM-default.yaml"



MINEDOJO_HEADLESS=1 python -u eval.py -log "CombatSpider-100k-seed1-JEPA-WM-default" -seed 1 -config "Experiments/eval/100K_CombatSpider_JEPA-WM-default.yaml" -mode reconstruction_clip
MINEDOJO_HEADLESS=1 python -u eval.py -log "CombatSpider-100k-seed1-JEPA_WM-pretirain" -seed 1 -config "Experiments/eval/100K_CombatSpider_JEPA-WM-pretirain.yaml" -mode reconstruction_clip
MINEDOJO_HEADLESS=1 python -u eval.py -log "CombatSpider-100k-seed1-JEPA_WM-symlog" -seed 1 -config "Experiments/eval/100K_CombatSpider_JEPA-WM-symlog.yaml" -mode reconstruction_clip
MINEDOJO_HEADLESS=1 python -u eval.py -log "CombatSpider-100k-seed1-STORM-default" -seed 1 -config "Experiments/eval/100K_CombatSpider_STORM-default.yaml" -mode reconstruction_clip
MINEDOJO_HEADLESS=1 python -u eval.py -log "CombatSpider-100k-seed1-JSTORM-symlog" -seed 1 -config "Experiments/eval/100K_CombatSpider_STORM-symlog.yaml" -mode reconstruction_clip


MINEDOJO_HEADLESS=1 python -u eval.py -log "CombatSpider-100k-seed1-JEPA-WM-default" -seed 1 -config "Experiments/eval/100K_CombatSpider_JEPA-WM-default.yaml" -mode agent
MINEDOJO_HEADLESS=1 python -u eval.py -log "CombatSpider-100k-seed1-JEPA_WM-pretirain" -seed 1 -config "Experiments/eval/100K_CombatSpider_JEPA-WM-pretirain.yaml" -mode agent
MINEDOJO_HEADLESS=1 python -u eval.py -log "CombatSpider-100k-seed1-JEPA_WM-symlog" -seed 1 -config "Experiments/eval/100K_CombatSpider_JEPA-WM-symlog.yaml" -mode agent
MINEDOJO_HEADLESS=1 python -u eval.py -log "CombatSpider-100k-seed1-STORM-default" -seed 1 -config "Experiments/eval/100K_CombatSpider_STORM-default.yaml" -mode agent
MINEDOJO_HEADLESS=1 python -u eval.py -log "CombatSpider-100k-seed1-JSTORM-symlog" -seed 1 -config "Experiments/eval/100K_CombatSpider_STORM-symlog.yaml" -mode agent

# MINEDOJO_HEADLESS=1 python -u eval.py -log "Tuning/${env_name}-100k-seed1-JEPA-WM-default" -seed 1 -config "Experiments/HuntCow/100K_HuntCow_JEPA-WM-default.yaml" -mode reconstruction_clip
# MINEDOJO_HEADLESS=1 python -u eval.py -log "Tuning/${env_name}-100k-seed1-JEPA-WM-pretirain" -seed 1 -config "Experiments/HuntCow/100K_HuntCow_JEPA-WM-pretirain.yaml" -mode reconstruction_clip
# # MINEDOJO_HEADLESS=1 python -u eval.py -log "Tuning/${env_name}-100k-seed1-JEPA-WM-symlog" -seed 1 -config "Experiments/HuntCow/100K_HuntCow_JEPA-WM-symlog.yaml" -mode reconstruction_clip
# MINEDOJO_HEADLESS=1 python -u eval.py -log "Tuning/${env_name}-100k-seed1-STORM-default" -seed 1 -config "Experiments/HuntCow/100K_HuntCow_STORM-default.yaml" -mode reconstruction_clip
# MINEDOJO_HEADLESS=1 python -u eval.py -log "Tuning/${env_name}-100k-seed1-STORM-symlog" -seed 1 -config "Experiments/HuntCow/100K_HuntCow_STORM-symlog.yaml" -mode reconstruction_clip

# MINEDOJO_HEADLESS=1 python -u eval.py -log "Tuning/${env_name}-100k-seed1-JEPA-WM-default" -seed 1 -config "Experiments/HuntCow/100K_HuntCow_JEPA-WM-default.yaml" -mode agent
# MINEDOJO_HEADLESS=1 python -u eval.py -log "Tuning/${env_name}-100k-seed1-JEPA-WM-pretirain" -seed 1 -config "Experiments/HuntCow/100K_HuntCow_JEPA-WM-pretirain.yaml" -mode agent
# # MINEDOJO_HEADLESS=1 python -u eval.py -log "Tuning/${env_name}-100k-seed1-JEPA-WM-symlog" -seed 1 -config "Experiments/HuntCow/100K_HuntCow_JEPA-WM-symlog.yaml" -mode agent
# MINEDOJO_HEADLESS=1 python -u eval.py -log "Tuning/${env_name}-100k-seed1-STORM-default" -seed 1 -config "Experiments/HuntCow/100K_HuntCow_STORM-default.yaml" -mode agent
# MINEDOJO_HEADLESS=1 python -u eval.py -log "Tuning/${env_name}-100k-seed1-STORM-symlog" -seed 1 -config "Experiments/HuntCow/100K_HuntCow_STORM-symlog.yaml" -mode agent