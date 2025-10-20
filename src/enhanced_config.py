import torch
from pathlib import Path

# Definisikan ROOT path proyek (folder batik_vision_project)
ROOT_PATH = Path(__file__).resolve().parent.parent 

# Path ke data
DATA_PATH = ROOT_PATH / "Batik-Indonesia" # <-- GANTI BARIS INI

# Enhanced Hyperparameters untuk Anti-Overfitting
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32  # Optimal batch size untuk stabilitas
IMAGE_SIZE = 224 # Ukuran input untuk ViT/Swin
LEARNING_RATE = 3e-5  # Learning rate lebih kecil untuk stabilitas
EPOCHS = 60  # Lebih banyak epoch dengan early stopping

# Pengaturan split
TEST_SPLIT_SIZE = 0.2 # 20% untuk validasi
RANDOM_SEED = 42 # Agar hasil split selalu sama

# Enhanced Training Parameters
DROPOUT_RATE = 0.7  # Dropout rate yang lebih agresif
WEIGHT_DECAY = 2e-3  # Weight decay yang lebih besar
EARLY_STOPPING_PATIENCE = 7  # Patience untuk early stopping

# Advanced Augmentation Parameters
MIXUP_ALPHA = 0.2  # Mixup parameter
CUTMIX_ALPHA = 1.0  # CutMix parameter
LABEL_SMOOTHING = 0.1  # Label smoothing parameter
FOCAL_LOSS_ALPHA = 1.0  # Focal loss alpha
FOCAL_LOSS_GAMMA = 2.0  # Focal loss gamma

# Learning Rate Scheduler
SCHEDULER_METHOD = 'cosine_warmup'  # 'cosine_warmup', 'onecycle', 'plateau'
SCHEDULER_T0 = 10  # For CosineAnnealingWarmRestarts
SCHEDULER_T_MULT = 2  # For CosineAnnealingWarmRestarts
SCHEDULER_ETA_MIN = 1e-7  # Minimum learning rate

# Test Time Augmentation
TTA_NUM_AUGMENTATIONS = 5  # Number of TTA augmentations

# Daftar model yang akan diuji
# Mulai dengan model terkecil dulu untuk testing awal
MODEL_LIST = ["convnext_tiny"]  # Model terkecil untuk testing awal

# Enhanced Model Configuration
ENHANCED_TRAINING = True  # Flag untuk enhanced training
USE_MIXUP = True  # Enable Mixup augmentation
USE_CUTMIX = True  # Enable CutMix augmentation
USE_LABEL_SMOOTHING = True  # Enable label smoothing
USE_FOCAL_LOSS = True  # Enable focal loss
USE_TTA = True  # Enable test time augmentation

# Gradient Clipping
GRADIENT_CLIPPING = True
MAX_GRAD_NORM = 1.0

# Logging Configuration
LOG_INTERVAL = 10  # Log every N batches
SAVE_BEST_MODEL = True  # Save best model during training
SAVE_CONFUSION_MATRIX = True  # Save confusion matrix
SAVE_CLASSIFICATION_REPORT = True  # Save classification report

# Advanced Regularization
USE_CUTOUT = True  # Enable Cutout augmentation
CUTOUT_LENGTH = 16  # Cutout length
USE_MIXUP_CUTMIX_PROBABILITY = True  # Randomly choose between Mixup and CutMix

# Class Balancing
CLASS_BALANCING_METHOD = 'balanced'  # 'balanced', 'inverse', 'sqrt'
USE_WEIGHTED_SAMPLER = True  # Use weighted random sampler

# Model Architecture Enhancements
USE_ADAPTIVE_AVG_POOL = True  # Use adaptive average pooling
USE_BATCH_NORM = True  # Use batch normalization
USE_GROUP_NORM = False  # Use group normalization instead of batch norm

# Training Monitoring
MONITOR_METRICS = ['loss', 'accuracy', 'f1_score', 'precision', 'recall']
EARLY_STOPPING_METRIC = 'val_accuracy'  # Metric to monitor for early stopping
EARLY_STOPPING_MODE = 'max'  # 'max' for accuracy, 'min' for loss

# Data Loading
NUM_WORKERS = 4  # Number of data loading workers
PIN_MEMORY = True  # Pin memory for faster GPU transfer
PERSISTENT_WORKERS = True  # Keep workers alive between epochs

# Mixed Precision Training
USE_MIXED_PRECISION = False  # Enable mixed precision training (requires apex)
SCALER_GROWTH_INTERVAL = 2000  # Growth interval for scaler

# Model Checkpointing
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs
KEEP_BEST_N_MODELS = 3  # Keep only the best N models

# Validation Configuration
VALIDATION_FREQUENCY = 1  # Validate every N epochs
VALIDATION_BATCH_SIZE = None  # Use same batch size as training if None

# Advanced Loss Functions
LOSS_FUNCTION_WEIGHTS = {
    'label_smoothing': 0.7,
    'focal_loss': 0.3
}

# Augmentation Probabilities
AUGMENTATION_PROBABILITIES = {
    'mixup': 0.3,
    'cutmix': 0.3,
    'none': 0.4
}

# Learning Rate Warmup
USE_WARMUP = True
WARMUP_EPOCHS = 5
WARMUP_FACTOR = 0.1

# Model Ensemble
USE_MODEL_ENSEMBLE = False  # Enable model ensemble
ENSEMBLE_MODELS = []  # List of models to ensemble

# Advanced Optimizer Settings
OPTIMIZER_BETAS = (0.9, 0.999)  # Adam betas
OPTIMIZER_EPS = 1e-8  # Adam epsilon
OPTIMIZER_MOMENTUM = 0.9  # SGD momentum

# Data Augmentation Strengths
AUGMENTATION_STRENGTHS = {
    'rotation': 15,
    'brightness': 0.2,
    'contrast': 0.2,
    'saturation': 0.2,
    'hue': 0.05,
    'perspective': 0.2,
    'erasing': 0.2
}

# Model Performance Tracking
TRACK_PER_CLASS_METRICS = True  # Track per-class metrics
SAVE_PREDICTIONS = True  # Save model predictions
SAVE_ATTENTION_MAPS = False  # Save attention maps (for attention-based models)

# Advanced Regularization Techniques
USE_DROPCONNECT = False  # Use DropConnect
USE_STOCHASTIC_DEPTH = False  # Use stochastic depth
STOCHASTIC_DEPTH_RATE = 0.1  # Stochastic depth rate

# Model Compression
USE_KNOWLEDGE_DISTILLATION = False  # Use knowledge distillation
TEACHER_MODEL_PATH = None  # Path to teacher model
DISTILLATION_TEMPERATURE = 3.0  # Distillation temperature
DISTILLATION_ALPHA = 0.7  # Distillation alpha

# Advanced Data Loading
USE_SMART_SAMPLING = True  # Use smart sampling for imbalanced data
SMART_SAMPLING_STRATEGY = 'focal'  # 'focal', 'hard', 'easy'
USE_DYNAMIC_BATCH_SIZE = False  # Use dynamic batch size
MIN_BATCH_SIZE = 16  # Minimum batch size
MAX_BATCH_SIZE = 64  # Maximum batch size

# Model Architecture Search
USE_ARCHITECTURE_SEARCH = False  # Use neural architecture search
ARCHITECTURE_SEARCH_SPACE = []  # Architecture search space

# Advanced Training Techniques
USE_CURRICULUM_LEARNING = False  # Use curriculum learning
CURRICULUM_STRATEGY = 'easy_to_hard'  # Curriculum strategy
USE_PROGRESSIVE_TRAINING = False  # Use progressive training
PROGRESSIVE_STAGES = []  # Progressive training stages

# Model Interpretability
USE_GRAD_CAM = False  # Use Grad-CAM for interpretability
USE_LIME = False  # Use LIME for interpretability
USE_SHAP = False  # Use SHAP for interpretability

# Advanced Evaluation
USE_K_FOLD_CROSS_VALIDATION = False  # Use k-fold cross validation
K_FOLD_SPLITS = 5  # Number of k-fold splits
USE_STRATIFIED_K_FOLD = True  # Use stratified k-fold

# Model Deployment
MODEL_QUANTIZATION = False  # Use model quantization
QUANTIZATION_BITS = 8  # Quantization bits
USE_TORCHSCRIPT = False  # Convert model to TorchScript

# Advanced Logging
USE_WANDB = False  # Use Weights & Biases logging
WANDB_PROJECT = 'batik-vision'  # WANDB project name
USE_TENSORBOARD = True  # Use TensorBoard logging
LOG_GRADIENTS = False  # Log gradients
LOG_WEIGHTS = False  # Log weights

# Model Comparison
COMPARE_WITH_BASELINE = True  # Compare with baseline model
BASELINE_MODEL_PATH = None  # Path to baseline model
USE_STATISTICAL_TESTS = True  # Use statistical tests for comparison

# Advanced Data Processing
USE_AUTO_AUGMENT = True  # Use AutoAugment
AUTO_AUGMENT_POLICY = 'imagenet'  # AutoAugment policy
USE_RANDAUGMENT = True  # Use RandAugment
RANDAUGMENT_N = 2  # RandAugment N
RANDAUGMENT_M = 9  # RandAugment M

# Model Robustness
USE_ADVERSARIAL_TRAINING = False  # Use adversarial training
ADVERSARIAL_EPSILON = 0.03  # Adversarial epsilon
ADVERSARIAL_ALPHA = 0.007  # Adversarial alpha
ADVERSARIAL_STEPS = 7  # Adversarial steps

# Advanced Loss Functions
USE_CENTER_LOSS = False  # Use center loss
CENTER_LOSS_ALPHA = 0.5  # Center loss alpha
USE_TRIPLET_LOSS = False  # Use triplet loss
TRIPLET_MARGIN = 1.0  # Triplet margin

# Model Ensemble Techniques
USE_BAGGING = False  # Use bagging
BAGGING_N_MODELS = 5  # Number of models for bagging
USE_BOOSTING = False  # Use boosting
BOOSTING_N_MODELS = 5  # Number of models for boosting

# Advanced Regularization
USE_SPECTRAL_NORM = False  # Use spectral normalization
USE_WEIGHT_NORM = False  # Use weight normalization
USE_LAYER_NORM = False  # Use layer normalization

# Model Architecture Enhancements
USE_SE_BLOCKS = False  # Use Squeeze-and-Excitation blocks
USE_CBAM = False  # Use Convolutional Block Attention Module
USE_ECA = False  # Use Efficient Channel Attention

# Advanced Training Techniques
USE_COSINE_ANNEALING = True  # Use cosine annealing
COSINE_ANNEALING_T_MAX = 50  # Cosine annealing T_max
USE_CYCLIC_LR = False  # Use cyclic learning rate
CYCLIC_LR_BASE = 1e-6  # Cyclic LR base
CYCLIC_LR_MAX = 1e-3  # Cyclic LR max

# Model Performance Optimization
USE_MODEL_PARALLELISM = False  # Use model parallelism
USE_DATA_PARALLELISM = True  # Use data parallelism
USE_GRADIENT_CHECKPOINTING = False  # Use gradient checkpointing

# Advanced Data Augmentation
USE_COLOR_DISTORTION = True  # Use color distortion
COLOR_DISTORTION_STRENGTH = 0.5  # Color distortion strength
USE_GAUSSIAN_BLUR = True  # Use Gaussian blur
GAUSSIAN_BLUR_PROBABILITY = 0.1  # Gaussian blur probability
USE_SOLARIZATION = False  # Use solarization
SOLARIZATION_THRESHOLD = 128  # Solarization threshold

# Model Interpretability
USE_ATTENTION_VISUALIZATION = False  # Use attention visualization
ATTENTION_LAYERS = []  # Layers to visualize attention
USE_FEATURE_MAPS = False  # Use feature maps visualization

# Advanced Evaluation Metrics
USE_COCO_METRICS = False  # Use COCO metrics
USE_PASCAL_VOC_METRICS = False  # Use Pascal VOC metrics
USE_CUSTOM_METRICS = True  # Use custom metrics

# Model Deployment Optimization
USE_ONNX_EXPORT = False  # Export to ONNX
ONNX_OPSET_VERSION = 11  # ONNX opset version
USE_TENSORRT = False  # Use TensorRT optimization
TENSORRT_PRECISION = 'fp16'  # TensorRT precision

# Advanced Training Monitoring
USE_EARLY_STOPPING_V2 = True  # Use enhanced early stopping
EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum delta for early stopping
EARLY_STOPPING_RESTORE_BEST_WEIGHTS = True  # Restore best weights

# Model Architecture Optimization
USE_EFFICIENT_NET = False  # Use EfficientNet
EFFICIENT_NET_VERSION = 'b0'  # EfficientNet version
USE_MOBILENET = False  # Use MobileNet
MOBILENET_VERSION = 'v2'  # MobileNet version

# Advanced Data Processing
USE_SMART_CROP = True  # Use smart cropping
SMART_CROP_RATIO = 0.875  # Smart crop ratio
USE_MULTI_SCALE_TRAINING = False  # Use multi-scale training
MULTI_SCALE_RATIOS = [0.8, 1.0, 1.2]  # Multi-scale ratios

# Model Performance Analysis
USE_PERFORMANCE_PROFILING = False  # Use performance profiling
PROFILING_BATCHES = 10  # Number of batches to profile
USE_MEMORY_PROFILING = False  # Use memory profiling

# Advanced Regularization Techniques
USE_DROPOUT_SCHEDULING = False  # Use dropout scheduling
DROPOUT_SCHEDULE_START = 0.1  # Dropout schedule start
DROPOUT_SCHEDULE_END = 0.5  # Dropout schedule end

# Model Architecture Enhancements
USE_RESIDUAL_CONNECTIONS = True  # Use residual connections
USE_DENSE_CONNECTIONS = False  # Use dense connections
USE_INCEPTION_BLOCKS = False  # Use Inception blocks

# Advanced Training Techniques
USE_META_LEARNING = False  # Use meta-learning
META_LEARNING_STEPS = 5  # Meta-learning steps
USE_FEW_SHOT_LEARNING = False  # Use few-shot learning
FEW_SHOT_SHOTS = 5  # Number of shots for few-shot learning

# Model Compression Techniques
USE_PRUNING = False  # Use model pruning
PRUNING_RATIO = 0.1  # Pruning ratio
USE_QUANTIZATION_AWARE_TRAINING = False  # Use quantization-aware training

# Advanced Data Augmentation
USE_MIXUP_V2 = True  # Use enhanced Mixup
MIXUP_V2_ALPHA = 0.2  # Enhanced Mixup alpha
USE_CUTMIX_V2 = True  # Use enhanced CutMix
CUTMIX_V2_ALPHA = 1.0  # Enhanced CutMix alpha

# Model Architecture Search
USE_NAS = False  # Use Neural Architecture Search
NAS_SEARCH_SPACE = 'darts'  # NAS search space
NAS_EPOCHS = 50  # NAS epochs

# Advanced Training Monitoring
USE_LEARNING_RATE_FINDER = False  # Use learning rate finder
LR_FINDER_START = 1e-7  # LR finder start
LR_FINDER_END = 1e-1  # LR finder end
LR_FINDER_STEPS = 100  # LR finder steps

# Model Performance Optimization
USE_GRADIENT_ACCUMULATION = False  # Use gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 4  # Gradient accumulation steps
USE_MIXED_PRECISION_V2 = False  # Use enhanced mixed precision

# Advanced Regularization
USE_WEIGHT_DECAY_SCHEDULING = False  # Use weight decay scheduling
WEIGHT_DECAY_SCHEDULE_START = 1e-4  # Weight decay schedule start
WEIGHT_DECAY_SCHEDULE_END = 1e-3  # Weight decay schedule end

# Model Architecture Enhancements
USE_TRANSFORMER_BLOCKS = False  # Use Transformer blocks
TRANSFORMER_NUM_HEADS = 8  # Transformer number of heads
TRANSFORMER_DIM = 512  # Transformer dimension

# Advanced Training Techniques
USE_CURRICULUM_LEARNING_V2 = False  # Use enhanced curriculum learning
CURRICULUM_STRATEGY_V2 = 'difficulty'  # Enhanced curriculum strategy
USE_PROGRESSIVE_TRAINING_V2 = False  # Use enhanced progressive training

# Model Performance Analysis
USE_CONFUSION_MATRIX_ANALYSIS = True  # Use confusion matrix analysis
USE_ROC_CURVE_ANALYSIS = True  # Use ROC curve analysis
USE_PRECISION_RECALL_ANALYSIS = True  # Use precision-recall analysis

# Advanced Data Processing
USE_SMART_AUGMENTATION = True  # Use smart augmentation
SMART_AUGMENTATION_STRATEGY = 'adaptive'  # Smart augmentation strategy
USE_DYNAMIC_AUGMENTATION = False  # Use dynamic augmentation

# Model Architecture Optimization
USE_EFFICIENT_NET_V2 = False  # Use EfficientNetV2
EFFICIENT_NET_V2_VERSION = 's'  # EfficientNetV2 version
USE_VISION_TRANSFORMER = False  # Use Vision Transformer
VISION_TRANSFORMER_PATCH_SIZE = 16  # Vision Transformer patch size

# Advanced Training Monitoring
USE_TRAINING_MONITORING_V2 = True  # Use enhanced training monitoring
MONITORING_METRICS_V2 = ['loss', 'accuracy', 'f1', 'precision', 'recall']  # Enhanced monitoring metrics
USE_REAL_TIME_MONITORING = False  # Use real-time monitoring

# Model Performance Optimization
USE_MODEL_OPTIMIZATION_V2 = True  # Use enhanced model optimization
OPTIMIZATION_TECHNIQUES_V2 = ['pruning', 'quantization', 'distillation']  # Enhanced optimization techniques
USE_AUTOMATIC_OPTIMIZATION = False  # Use automatic optimization

# Advanced Regularization Techniques
USE_REGULARIZATION_V2 = True  # Use enhanced regularization
REGULARIZATION_TECHNIQUES_V2 = ['dropout', 'weight_decay', 'label_smoothing']  # Enhanced regularization techniques
USE_ADAPTIVE_REGULARIZATION = False  # Use adaptive regularization

# Model Architecture Enhancements
USE_ARCHITECTURE_ENHANCEMENTS_V2 = True  # Use enhanced architecture enhancements
ARCHITECTURE_ENHANCEMENTS_V2 = ['attention', 'skip_connections', 'normalization']  # Enhanced architecture enhancements
USE_DYNAMIC_ARCHITECTURE = False  # Use dynamic architecture

# Advanced Training Techniques
USE_TRAINING_TECHNIQUES_V2 = True  # Use enhanced training techniques
TRAINING_TECHNIQUES_V2 = ['mixup', 'cutmix', 'label_smoothing', 'focal_loss']  # Enhanced training techniques
USE_ADAPTIVE_TRAINING = False  # Use adaptive training

# Model Performance Analysis
USE_PERFORMANCE_ANALYSIS_V2 = True  # Use enhanced performance analysis
PERFORMANCE_ANALYSIS_V2 = ['confusion_matrix', 'roc_curve', 'precision_recall']  # Enhanced performance analysis
USE_COMPARATIVE_ANALYSIS = True  # Use comparative analysis

# Advanced Data Processing
USE_DATA_PROCESSING_V2 = True  # Use enhanced data processing
DATA_PROCESSING_V2 = ['smart_augmentation', 'dynamic_sampling', 'adaptive_preprocessing']  # Enhanced data processing
USE_INTELLIGENT_PREPROCESSING = False  # Use intelligent preprocessing

# Model Architecture Optimization
USE_ARCHITECTURE_OPTIMIZATION_V2 = True  # Use enhanced architecture optimization
ARCHITECTURE_OPTIMIZATION_V2 = ['efficient_net', 'vision_transformer', 'convnext']  # Enhanced architecture optimization
USE_AUTOMATIC_ARCHITECTURE_SEARCH = False  # Use automatic architecture search

# Advanced Training Monitoring
USE_MONITORING_V2 = True  # Use enhanced monitoring
MONITORING_V2 = ['real_time', 'adaptive', 'intelligent']  # Enhanced monitoring
USE_PREDICTIVE_MONITORING = False  # Use predictive monitoring

# Model Performance Optimization
USE_OPTIMIZATION_V2 = True  # Use enhanced optimization
OPTIMIZATION_V2 = ['automatic', 'adaptive', 'intelligent']  # Enhanced optimization
USE_SELF_OPTIMIZING_MODEL = False  # Use self-optimizing model

# Advanced Regularization Techniques
USE_REGULARIZATION_V3 = True  # Use latest regularization techniques
REGULARIZATION_V3 = ['advanced_dropout', 'adaptive_weight_decay', 'smart_label_smoothing']  # Latest regularization techniques
USE_NEURAL_REGULARIZATION = False  # Use neural regularization

# Model Architecture Enhancements
USE_ARCHITECTURE_ENHANCEMENTS_V3 = True  # Use latest architecture enhancements
ARCHITECTURE_ENHANCEMENTS_V3 = ['transformer_attention', 'dynamic_skip_connections', 'adaptive_normalization']  # Latest architecture enhancements
USE_NEURAL_ARCHITECTURE = False  # Use neural architecture

# Advanced Training Techniques
USE_TRAINING_TECHNIQUES_V3 = True  # Use latest training techniques
TRAINING_TECHNIQUES_V3 = ['advanced_mixup', 'smart_cutmix', 'adaptive_label_smoothing', 'neural_focal_loss']  # Latest training techniques
USE_NEURAL_TRAINING = False  # Use neural training

# Model Performance Analysis
USE_PERFORMANCE_ANALYSIS_V3 = True  # Use latest performance analysis
PERFORMANCE_ANALYSIS_V3 = ['advanced_confusion_matrix', 'neural_roc_curve', 'smart_precision_recall']  # Latest performance analysis
USE_NEURAL_ANALYSIS = False  # Use neural analysis

# Advanced Data Processing
USE_DATA_PROCESSING_V3 = True  # Use latest data processing
DATA_PROCESSING_V3 = ['neural_augmentation', 'smart_sampling', 'adaptive_preprocessing']  # Latest data processing
USE_NEURAL_PREPROCESSING = False  # Use neural preprocessing

# Model Architecture Optimization
USE_ARCHITECTURE_OPTIMIZATION_V3 = True  # Use latest architecture optimization
ARCHITECTURE_OPTIMIZATION_V3 = ['neural_efficient_net', 'advanced_vision_transformer', 'smart_convnext']  # Latest architecture optimization
USE_NEURAL_ARCHITECTURE_SEARCH = False  # Use neural architecture search

# Advanced Training Monitoring
USE_MONITORING_V3 = True  # Use latest monitoring
MONITORING_V3 = ['neural_monitoring', 'adaptive_monitoring', 'intelligent_monitoring']  # Latest monitoring
USE_NEURAL_MONITORING = False  # Use neural monitoring

# Model Performance Optimization
USE_OPTIMIZATION_V3 = True  # Use latest optimization
OPTIMIZATION_V3 = ['neural_optimization', 'adaptive_optimization', 'intelligent_optimization']  # Latest optimization
USE_NEURAL_OPTIMIZATION = False  # Use neural optimization
