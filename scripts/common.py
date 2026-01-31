# MODEL_CKPT = "best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr5e-5_bs32.pt"
MODEL_CKPT = "best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd45.pt"
# MODEL_CKPT = "unoptimized.best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd42.pt"
# MODEL_CKPT = "tdann.model_final_checkpoint_phase199_seed0.torch"
# MODEL_CKPT = "swapopt_single_sheet_seed0"

MODEL_CKPTS = [
    "best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd42.pt",
    "best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd43.pt",
    "best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd44.pt",
    "best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd45.pt",
    "best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd46.pt",
]

UNOPTIMIZED_CKPTS = [
    "unoptimized.best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd42.pt",
    "unoptimized.best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd43.pt",
    "unoptimized.best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd44.pt",
    "unoptimized.best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd45.pt",
    "unoptimized.best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd46.pt",
]

TDANN_CKPTS = [
    "tdann.model_final_checkpoint_phase199_seed0.torch",
    "tdann.model_final_checkpoint_phase199_seed1.torch",
    "tdann.model_final_checkpoint_phase199_seed2.torch",
    "tdann.model_final_checkpoint_phase199_seed3.torch",
    "tdann.model_final_checkpoint_phase199_seed4.torch",
]

SWAPOPT_CKPTS = [
    "swapopt_single_sheet_seed0",
    "swapopt_single_sheet_seed1",
    "swapopt_single_sheet_seed2",
    "swapopt_single_sheet_seed3",
    "swapopt_single_sheet_seed4",
]

SWAPOPT_ONELAYER_CKPTS = [
    "swapopt_seed0",
    "swapopt_seed1",
    "swapopt_seed2",
    "swapopt_seed3",
    "swapopt_seed4",
]

ONELAYER_CKPTS = [
    "best_transformed_model_global_vjepa_18_single_neighbInf_kinetics400_lr1e-4_bs32_sd42.pt",
    "best_transformed_model_global_vjepa_18_single_neighbInf_kinetics400_lr1e-4_bs32_sd43.pt",
    "best_transformed_model_global_vjepa_18_single_neighbInf_kinetics400_lr1e-4_bs32_sd44.pt",
    "best_transformed_model_global_vjepa_18_single_neighbInf_kinetics400_lr1e-4_bs32_sd45.pt",
    "best_transformed_model_global_vjepa_18_single_neighbInf_kinetics400_lr1e-4_bs32_sd46.pt",
]   

HUMAN_C = '#2E7D32'
MODEL_C = '#7C8DB0'
DEFAULT_C = 'gray'

LOCALIZER_P_THRESHOLD = 1e-3
LOCALIZER_T_THRESHOLD = 10
LOCALIZER_BIOMOTION_T_THRESHOLD = 0
LOCALIZER_FLOW_T_THRESHOLD = 30


# Colors

all_roi_colors = {
    "Faces_moving_localizer": ("moving-face", (0.91, 0.30, 0.24)),   # soft warm red
    "Bodies_moving_localizer": ("moving-body", (0.20, 0.63, 0.55)),  # cool turquoise
    "Scenes_moving_localizer": ("moving-place", (0.95, 0.77, 0.06)), # golden yellow
    "Faces_static_localizer": ("static-face", (1.00, 0.80, 0.80)),
    "Bodies_static_localizer": ("static-body", (0.80, 1.00, 0.80)),
    "Scenes_static_localizer": ("static-place", (1.00, 0.90, 0.70)),
    "Faces_static": ("static-face", (0.75, 0.00, 0.00)),
    "Bodies_static": ("static-body", (0.00, 0.45, 0.00)),
    "Scenes_static": ("static-place", (0.80, 0.35, 0.00)),
    "Faces_moving": ("dynamic-face", (1.00, 0.80, 0.80)),
    "Bodies_moving": ("dynamic-body", (0.80, 1.00, 0.80)),
    "Scenes_moving": ("dynamic-place", (1.00, 0.90, 0.70)),
    "object": ("object", (0.20, 0.20, 0.80)),
    "V6":    ("V6",  (0.00, 0.78, 0.88)),  # softer aqua, less neon, more elegant
    "MT-Huk": ("MT", (0.90, 0.25, 0.65)),  # smoother magenta, high-end look
    "pSTS": ("pSTS", (0.55, 0.45, 0.95)),   # soft lavender–violet
    "V6-enhanced": ("V6-enhanced", (0.00, 0.60, 0.70)),
    "pSTS-enhanced": ("pSTS-enhanced", (0.40, 0.30, 0.70)),
}

all_roi_colors['face'] = all_roi_colors['Faces_moving_localizer']
all_roi_colors['body'] = all_roi_colors['Bodies_moving_localizer']
all_roi_colors['place'] = all_roi_colors['Scenes_moving_localizer']
all_roi_colors['mt'] = all_roi_colors['MT-Huk']
all_roi_colors['v6'] = all_roi_colors['V6']
all_roi_colors['psts'] = all_roi_colors['pSTS']

roi_groups = {
    "face-response": ["Faces_static", "Faces_moving"],
    "body-response": ["Bodies_static", "Bodies_moving"],
    "place-response": ["Scenes_static", "Scenes_moving"],
    "face": ["Faces_static_localizer", "Faces_moving_localizer"],
    "body": ["Bodies_static_localizer", "Bodies_moving_localizer"],
    "place": ["Scenes_static_localizer", "Scenes_moving_localizer"],
    "motion": ["V6", "pSTS", "MT-Huk"],
    "motion2": ["V6", "pSTS"],
    "motion3": ["V6-enhanced", "pSTS-enhanced"],
    "V6": ["V6"],
    "MT": ["MT-Huk"],
    "pSTS": ["pSTS"],
    "categorical": [
        "Faces_static_localizer", "Bodies_static_localizer", "Scenes_static_localizer",
        "Faces_moving_localizer", "Bodies_moving_localizer", "Scenes_moving_localizer",
    ],
    "categorical2": [
        "Faces_moving_localizer", "Bodies_moving_localizer", "Scenes_moving_localizer",
    ],
    "fLoc": ["face", "body", "place", "object"],
    "fLoc2": ["face", "body", "place"],
    "all": list(all_roi_colors.keys()),
}