MODEL_CKPT = "best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr5e-5_bs32.pt"
# MODEL_CKPT = "best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd42.pt"
# MODEL_CKPT = "tdann.model_final_checkpoint_phase199_seed0.torch"
# MODEL_CKPT = "swapopt"

MODEL_CKPTS = [
    # "best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr5e-5_bs32.pt",
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
    "swapopt",
]

HUMAN_C = '#2E7D32'
MODEL_C = '#7C8DB0'
DEFAULT_C = 'gray'

LOCALIZER_P_THRESHOLD = 1e-3
LOCALIZER_T_THRESHOLD = 0
