# env_name=MsPacman
# python -u train.py \
#     -n "${env_name}-life_done-wm_2L512D8H-100k-seed1" \
#     -seed 1 \
#     -config_path "config_files/STORM.yaml" \
#     -env_name "ALE/${env_name}-v5" \
#     -trajectory_path "D_TRAJ/${env_name}.pkl" 

env_name=HuntCow #CombatSpider HuntCow
# MINEDOJO_HEADLESS=1 python -u train.py -log "${env_name}-100k-seed1-JEPA_WM" -seed 1 -config "config_files/100K_CombatSpider_JEPA-WM.yaml" 
# MINEDOJO_HEADLESS=1 python -u train.py -log "${env_name}-100k-seed1-STORM" -seed 1 -config "Experiments/100K_HuntCow_STORM.yaml" 
# MINEDOJO_HEADLESS=1 python -u train.py -log "${env_name}-100k-seed1-JEPA_WM-symlog" -seed 1 -config "Experiments/100K_CombatSpider_JEPA-WM-symlog.yaml"
# MINEDOJO_HEADLESS=1 python -u train.py -log "TuningExp/${env_name}-100k-seed1-JEPA_WM-default" -seed 1 -config "Experiments/100K_CombatSpider_JEPA-WM-default.yaml"
# MINEDOJO_HEADLESS=1 python -u train.py -log "ReconTest/${env_name}-100k-seed1-STORM" -seed 1 -config "Experiments/100K_CombatSpider_STORM.yaml"
# MINEDOJO_HEADLESS=1 python -u train.py -log "ReconTest/${env_name}-100k-seed1-JEPA_WM" -seed 1 -config "Experiments/100K_CombatSpider_JEPA-WM-recon.yaml"
# MINEDOJO_HEADLESS=1 python -u train.py -log "ReconTest/${env_name}-100k-seed1-JEPA_WM" -seed 1 -config "Experiments/100K_CombatSpider_JEPA-WM-recon&symlog.yaml"

MINEDOJO_HEADLESS=1 python -u train.py -log "Tuning/${env_name}-100k-seed1-JEPA-WM-default" -seed 1 -config "Experiments/HuntCow/100K_HuntCow_JEPA-WM-default.yaml"
MINEDOJO_HEADLESS=1 python -u train.py -log "Tuning/${env_name}-100k-seed1-JEPA-WM-pretirain" -seed 1 -config "Experiments/HuntCow/100K_HuntCow_JEPA-WM-pretirain.yaml"
MINEDOJO_HEADLESS=1 python -u train.py -log "Tuning/${env_name}-100k-seed1-JEPA-WM-symlog" -seed 1 -config "Experiments/HuntCow/100K_HuntCow_JEPA-WM-symlog.yaml"
MINEDOJO_HEADLESS=1 python -u train.py -log "Tuning/${env_name}-100k-seed1-STORM-default" -seed 1 -config "Experiments/HuntCow/100K_HuntCow_STORM-default.yaml"
MINEDOJO_HEADLESS=1 python -u train.py -log "Tuning/${env_name}-100k-seed1-STORM-symlog" -seed 1 -config "Experiments/HuntCow/100K_HuntCow_STORM-symlog.yaml"

# MINEDOJO_HEADLESS=1 python -u train.py -log "${env_name}-100k-seed1-JEPA_WM-pretirain-0" -seed 1 -config "Experiments/100K_CombatSpider_JEPA-WM-pretirain.yaml"

