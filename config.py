import os, torch, yaml, csv, json, inspect, shutil, argparse
from datetime import datetime

# --- Constants & Indices ---
class FeatureIndices:
    P_LOAD=0; Q_LOAD=1; P_EXT_GRID=2; Q_EXT_GRID=3; P_CONV=4; Q_CONV=5; P_REN=6; Q_REN=7; VM=8; VA=9
    FEATURE_NAMES = ['p_load', 'q_load', 'p_ext_grid', 'q_ext_grid', 'p_conv', 'q_conv', 'p_ren', 'q_ren', 'vm', 'va']
    NUM_FEATURES = 10

class TargetIndices:
    P_LOAD=0; Q_LOAD=1; P_EXT_GRID=2; Q_EXT_GRID=3; P_CONV=4; Q_CONV=5; P_REN=6; Q_REN=7; VM=8; VA=9
    NUM_TARGETS = 10

class ModelOutputIndices:
    P_LOAD=0; Q_LOAD=1; P_EXT_GRID=2; Q_EXT_GRID=3; P_CONV=4; Q_CONV=5; P_REN=6; Q_REN=7; VM=8; VA=9
    NUM_OUTPUTS = 10

# --- Configuration ---
class Config:
    """Central configuration. Strict YAML loading required."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    EXPERIMENTAL_RESULTS_DIR = os.path.join(ROOT_DIR, 'experimental_results')
    _CURRENT_RUN_TIMESTAMP = None

    MODELS_TO_TEST = ['GCN', 'adaptiveGCN', 'AdaptivePIGCN', 'PIGCLSTM', 'PIGCGRU', 'ResnetPIGCLSTM', 'ResnetPIGCGRU']
    MODEL_TEST_CONFIGS = {
        'quick': ['AdaptivePIGCN'], 'core': ['adaptiveGCN', 'AdaptivePIGCN'],
        'comprehensive': ['GCN', 'adaptiveGCN', 'AdaptivePIGCN', 'PIGCLSTM', 'PIGCGRU'],
        'physics_only': ['AdaptivePIGCN', 'PIGCLSTM', 'PIGCGRU', 'ResnetPIGCLSTM', 'ResnetPIGCGRU'],
        'non_physics_only': ['GCN', 'adaptiveGCN'],
        'sequential_only': ['PIGCLSTM', 'PIGCGRU', 'ResnetPIGCLSTM', 'ResnetPIGCGRU'],
        'all': ['GCN', 'adaptiveGCN', 'AdaptivePIGCN', 'PIGCLSTM', 'PIGCGRU', 'ResnetPIGCLSTM', 'ResnetPIGCGRU']
    }

    class _ModelConfig:
        INPUT_DIM=10; OUTPUT_DIM=10; FEATURE_DIM=10; DROPOUT=0.1
        
        @staticmethod
        def _get_setting(num_buses, normal, medium, large, config_instance=None):
            # Look up setting on the instance if provided, otherwise default to 'normal'
            cap = 'normal'
            if config_instance:
                cap = getattr(config_instance, f'CAPACITY_{num_buses}_BUS', 'normal')
            return {'normal': normal, 'medium': medium, 'large': large}.get(cap, normal)
            
        @classmethod
        def get_hidden_dim_range(cls, nb, config_instance=None): return cls._get_setting(nb, (64, 128), (96, 192), (64, 256), config_instance)
        @classmethod
        def get_num_gc_layers_range(cls, nb, config_instance=None): return cls._get_setting(nb, (1, 2), (2, 3), (2, 3), config_instance)
        @classmethod
        def get_embedding_dim_range(cls, nb, config_instance=None): return cls._get_setting(nb, (64, 150), (100, 200), (100, 300), config_instance)
        
        @classmethod
        def get_phi_range(cls, nb): return (0.0, 1.0)
        
        @classmethod
        def get_sequential_ranges(cls, nb):
            if nb <= 33: return {'hidden_dim': (32, 64), 'sequence_length': (5, 10), 'rnn_layers': (1, 3)}
            if nb <= 57: return {'hidden_dim': (16, 48), 'sequence_length': (3, 8), 'rnn_layers': (1, 2)}
            return {'hidden_dim': (16, 32), 'sequence_length': (3, 5), 'rnn_layers': (1, 2)}

        @staticmethod
        def get_recommended_model(nb): return "PIGCGRU" if nb <= 33 else "AdaptivePIGCN"
        
        @staticmethod
        def get_adaptive_mosoa_params(nb, config_instance=None):
            # Strict config enforcement - no fallback defaults
            if config_instance:
                num_seagulls = getattr(config_instance, f'OPTIMIZATION_CASE{nb}_NUM_SEAGULLS', None)
                max_iterations = getattr(config_instance, f'OPTIMIZATION_CASE{nb}_MAX_ITERATIONS', None)
                strategy = getattr(config_instance, f'OPTIMIZATION_CASE{nb}_STRATEGY', 'custom')
                
                if num_seagulls is not None and max_iterations is not None:
                     return {
                        'num_seagulls': num_seagulls,
                        'max_iterations': max_iterations,
                        'strategy': strategy,
                        'description': f'Configured via config.yaml ({strategy})'
                    }
            
            # If we reach here, it means configuration is missing
            raise AttributeError(f"Missing optimization configuration for case{nb} in config.yaml. "
                               f"Please ensure OPTIMIZATION_CASE{nb}_NUM_SEAGULLS and OPTIMIZATION_CASE{nb}_MAX_ITERATIONS are set.")

    # Single Unified Config Class - No more subclasses
    # Individual instances can still override specific attributes if absolutely necessary
    # but the base class provides EVERYTHING needed for consistency.
    
    GCNConfig = _ModelConfig()
    
    # Adaptive GCN just needs the base config (it will use PHI and EMBEDDING_DIM from base)
    adaptiveGCNConfig = _ModelConfig()
    
    # AdaptivePIGCN overrides one specific range (GC Layers)
    AdaptivePIGCNConfig = _ModelConfig()
    
    # Sequential models just need the base config (they will use get_sequential_ranges from base)
    # We set DROPOUT=0.25 as the only override
    PIGCLSTMConfig = _ModelConfig()
    PIGCGRUConfig = _ModelConfig()
    ResnetPIGCGRUConfig = _ModelConfig()
    ResnetPIGCLSTMConfig = _ModelConfig()

    @property
    def CURRENT_RUN_DIR(self): return os.path.join(self.EXPERIMENTAL_RESULTS_DIR, f'run_{self._CURRENT_RUN_TIMESTAMP}')
    @property
    def model_config_map(self): return {'GCN': self.GCNConfig, 'adaptiveGCN': self.adaptiveGCNConfig, 'AdaptivePIGCN': self.AdaptivePIGCNConfig, 'PIGCLSTM': self.PIGCLSTMConfig, 'PIGCGRU': self.PIGCGRUConfig, 'ResnetPIGCGRU': self.ResnetPIGCGRUConfig, 'ResnetPIGCLSTM': self.ResnetPIGCLSTMConfig}

    def __init__(self, data_mode='train', train_timesteps=None, test_timesteps=None, clear_results=False, hours_per_day=24, sequence_length=None, yaml_config_path=None, load_yaml=True, cli_args=None):
        if not load_yaml: raise ValueError("load_yaml=False not allowed.")
        yaml_path = yaml_config_path or (cli_args.config if cli_args else 'config.yaml')
        yaml_full_path = yaml_path if os.path.isabs(yaml_path) else os.path.join(self.ROOT_DIR, yaml_path)
        if not os.path.exists(yaml_full_path): raise FileNotFoundError(f"Missing config: {yaml_full_path}")
        
        with open(yaml_full_path) as f: self._merge_yaml(yaml.safe_load(f))
        
        if cli_args:
            if cli_args.models:
                if cli_args.models in self.MODEL_TEST_CONFIGS: setattr(Config, 'test_config', cli_args.models); setattr(Config, 'models_to_train', "all")
                else: setattr(Config, 'models_to_train', [m.strip() for m in cli_args.models.split(',')])
            if cli_args.buses: setattr(Config, 'bus_systems', [33, 57, 118] if cli_args.buses.lower() == 'all' else [int(b.strip()) for b in cli_args.buses.split(',')])
            if cli_args.timesteps: train_timesteps = test_timesteps = cli_args.timesteps
            if cli_args.mode: data_mode = cli_args.mode
            if cli_args.clear_results is not None: clear_results = cli_args.clear_results
            
        if hasattr(Config, 'clear_results'): clear_results = Config.clear_results
        # Check instance attribute (from YAML) if not found on class
        if hasattr(self, 'clear_results'): clear_results = self.clear_results

        if clear_results and os.path.exists(self.EXPERIMENTAL_RESULTS_DIR): shutil.rmtree(self.EXPERIMENTAL_RESULTS_DIR, ignore_errors=True)
        
        self.DATA_MODE_TIMESTEPS = {'train': train_timesteps or getattr(self, 'train_timesteps'), 'test': test_timesteps or getattr(self, 'test_timesteps')}
        self.DATA_MODE = data_mode
        self.HOURS_PER_DAY = hours_per_day
        
        # Only set SEQUENCE_LENGTH if explicitly provided or found in YAML (via _merge_yaml)
        # NO FALLBACK allowed here.
        if sequence_length is not None:
            self.SEQUENCE_LENGTH = sequence_length
        # If it wasn't in YAML and wasn't in args, it remains unset.
        # Accessing it later will raise AttributeError, satisfying "fail fast".
        
        self.DATA_DIR = os.path.join(self.ROOT_DIR, 'data', data_mode)
        
        # Determine debug settings
        if data_mode == 'train': self.DEBUG_ENABLE = getattr(self, 'DEBUG_TRAIN_ENABLE', False); self.SHOW_DETAILED_PROGRESS = getattr(self, 'DEBUG_TRAIN_SHOW_PROGRESS', False)
        else: self.DEBUG_ENABLE = getattr(self, 'DEBUG_TEST_ENABLE', True); self.SHOW_DETAILED_PROGRESS = getattr(self, 'DEBUG_TEST_SHOW_PROGRESS', True)
        
        # Test mode check
        test_mode = any('test' in (f[0].f_globals.get('__file__', '') or '') for f in inspect.stack())
        if not test_mode: self._initialize_run_timestamp()
        else: self._CURRENT_RUN_TIMESTAMP = self._CURRENT_RUN_TIMESTAMP or 'test_mode'

    def _merge_yaml(self, yml):
        flat = {}
        def _flatten(d, p=''):
            for k, v in d.items():
                if isinstance(v, dict): _flatten(v, f"{p}_{k}" if p else k)
                else: flat[f"{p}_{k}" if p else k] = v
        _flatten(yml)
        
        key_map = {
            'system_device': 'DEVICE', 'system_num_buses': 'NUM_BUSES', 'system_test_cases': 'TEST_CASES', 'system_seed': 'SEED', 'system_case_name': 'CASE_NAME',
            'training_batch_size': 'BATCH_SIZE', 'training_learning_rate': 'LEARNING_RATE', 'training_max_grad_norm': 'MAX_GRAD_NORM', 'training_num_epochs': 'NUM_EPOCHS',
            'training_early_stopping_patience': 'EARLY_STOPPING_PATIENCE', 'training_gradient_accumulation_steps': 'GRADIENT_ACCUMULATION_STEPS',
            'training_use_learning_rate_scheduler': 'USE_LEARNING_RATE_SCHEDULER', 'training_cosine_annealing_lr_t_max': 'COSINEANNEALINGLR_T_MAX',
            'training_cosine_annealing_lr_eta_min': 'COSINEANNEALINGLR_ETA_MIN', 'training_weight_decay': 'WEIGHT_DECAY',
            'physics_split_mode': 'DATA_SPLIT_MODE', 'physics_splits_train': 'TRAIN_SPLIT', 'physics_splits_val': 'VAL_SPLIT',
            'data_hours_per_day': 'HOURS_PER_DAY', 'data_sequence_length': 'SEQUENCE_LENGTH', 'data_contingency_rate': 'CONTINGENCY_RATE', 'data_pmu_coverage': 'PMU_COVERAGE',
            'moopf_weights_loss': 'MOOPF_WEIGHT_LOSS', 'moopf_weights_voltage_deviation': 'MOOPF_WEIGHT_VDEV', 'moopf_weights_carbon': 'MOOPF_WEIGHT_CARBON',
            'experimental_test_config': 'test_config', 'experimental_bus_systems': 'bus_systems', 'experimental_models_to_train': 'models_to_train',
            'experimental_data_mode': 'data_mode', 'experimental_train_timesteps': 'train_timesteps', 'experimental_test_timesteps': 'test_timesteps',
            'experimental_parallel_data_loading': 'parallel_data_loading', 'experimental_data_workers': 'NUM_WORKERS', 'experimental_pin_memory': 'PIN_MEMORY',
            'experimental_gradient_checkpointing': 'GRADIENT_CHECKPOINTING', 'experimental_save_checkpoints': 'SAVE_CHECKPOINTS', 'experimental_enable_logging': 'LOGGING_ENABLED', 'experimental_clear_results': 'clear_results',
            'optimization_case33_num_seagulls': 'OPTIMIZATION_CASE33_NUM_SEAGULLS', 'optimization_case33_max_iterations': 'OPTIMIZATION_CASE33_MAX_ITERATIONS',
            'optimization_case57_num_seagulls': 'OPTIMIZATION_CASE57_NUM_SEAGULLS', 'optimization_case57_max_iterations': 'OPTIMIZATION_CASE57_MAX_ITERATIONS',
            'optimization_case118_num_seagulls': 'OPTIMIZATION_CASE118_NUM_SEAGULLS', 'optimization_case118_max_iterations': 'OPTIMIZATION_CASE118_MAX_ITERATIONS',
            'debug_train_show_detailed_progress': 'DEBUG_TRAIN_SHOW_PROGRESS', 'debug_test_show_detailed_progress': 'DEBUG_TEST_SHOW_PROGRESS', 'debug_log_interval': 'DEBUG_LOG_INTERVAL'
        }
        for k, v in flat.items():
            if v is None: continue
            
            # Robust number parsing handling scientific notation (e.g., 1e-4)
            if isinstance(v, str):
                try:
                    f_val = float(v)
                    if f_val.is_integer() and '.' not in v and 'e' not in v.lower():
                        v = int(v)
                    else:
                        v = f_val
                except ValueError:
                    pass

            attr = key_map.get(k, k.upper().replace('-', '_'))
            
            # Set attributes on the instance (self), not the class (Config)
            # This prevents global state pollution and race conditions in parallel runs
            target = self
            setattr(target, attr, v)
            
            # Store capacity settings on instance as well
            if 'model_capacity_bus_' in k: 
                setattr(self, f'CAPACITY_{k.split("_")[-1]}_BUS', v)
            
        if hasattr(self, 'CASE_NAME') and self.CASE_NAME and 'system_limits' in yml:
            lim = yml['system_limits'].get(self.CASE_NAME.lower(), {})
            for k, v in {'base_mva': 'BASE_MVA', 'v_min': 'V_MIN', 'v_max': 'V_MAX'}.items():
                if k in lim: setattr(self, v, lim[k])

    def create_run_directories(self):
        for d in [self.DATA_DIR, self.EXPERIMENTAL_RESULTS_DIR, self.CURRENT_RUN_DIR]: os.makedirs(d, exist_ok=True)
        self._update_latest_run_link(); self._create_run_metadata()

    @staticmethod
    def get_model_class_map():
        from models.adaptive_gcn import AdaptiveGCN
        from models.gcn import GCN
        from models.graph_rnn import PIGCLSTM, PIGCGRU, ResnetPIGCGRU, ResnetPIGCLSTM
        return {'adaptiveGCN': AdaptiveGCN, 'GCN': GCN, 'AdaptivePIGCN': AdaptiveGCN, 'PIGCLSTM': PIGCLSTM, 'PIGCGRU': PIGCGRU, 'ResnetPIGCGRU': ResnetPIGCGRU, 'ResnetPIGCLSTM': ResnetPIGCLSTM}

    @staticmethod
    def get_models_to_test(cfg='quick'): return Config.MODEL_TEST_CONFIGS.get(cfg, Config.MODEL_TEST_CONFIGS['quick'])
    @staticmethod
    def is_sequential_model(name): return 'LSTM' in name.upper() or 'GRU' in name.upper()
    @staticmethod
    def is_physics_informed(name): return 'PI' in name
    @staticmethod
    def uses_adaptive_graph(name): return name in ['PIGCLSTM', 'PIGCGRU', 'adaptiveGCN', 'AdaptivePIGCN', 'ResnetPIGCGRU', 'ResnetPIGCLSTM']

    def get_model_eval_dir(self, nb, name): return os.path.join(self.CURRENT_RUN_DIR, f"{nb}bus", "models", name)
    def get_renewable_impacts_dir(self, nb, name): return os.path.join(self.get_model_eval_dir(nb, name), "renewable_impacts")
    def get_model_checkpoint_path(self, nb, name): return os.path.join(self.get_model_eval_dir(nb, name), "checkpoint.pth")
    def get_moopf_results_path(self, nb, name): return os.path.join(self.get_model_eval_dir(nb, name), "moopf_results.csv")
    def get_convergence_plot_path(self, nb, name): return os.path.join(self.get_model_eval_dir(nb, name), "mosoa_conv.png")
    def get_training_history_path(self, nb, name): return os.path.join(self.get_model_eval_dir(nb, name), "train_hist.png")
    def get_summary_path(self, nb, name): return os.path.join(self.get_model_eval_dir(nb, name), "summary.csv")
    
    def get_training_log_path(self, nb, name, mode=None):
        log_dir = os.path.join(self.CURRENT_RUN_DIR, f"{nb}bus", "log"); os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, f"{name}_{mode or getattr(self, 'DATA_MODE', 'train')}.log")

    @staticmethod
    def parse_cli_args():
        p = argparse.ArgumentParser(description='Physics-Informed Machine Learning')
        p.add_argument('--models', type=str); p.add_argument('--buses', type=str); p.add_argument('--mode', choices=['train', 'test'])
        p.add_argument('--timesteps', type=int); p.add_argument('--config', default='config.yaml'); p.add_argument('--clear_results', action='store_true', default=None)
        p.add_argument('--no_clear_results', action='store_false', dest='clear_results'); p.add_argument('--no_progress_bar', action='store_true')
        return p.parse_args()

    def _initialize_run_timestamp(self): self._CURRENT_RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def _update_latest_run_link(self):
        os.makedirs(self.EXPERIMENTAL_RESULTS_DIR, exist_ok=True)
        # Rule #2: No silent warning. If we can't write metadata, it's a permission/disk error.
        with open(os.path.join(self.EXPERIMENTAL_RESULTS_DIR, 'latest_run_info.txt'), 'w') as f:
            f.write(f"Latest run: run_{self._CURRENT_RUN_TIMESTAMP}\nDirectory: {self.CURRENT_RUN_DIR}\nStarted: {datetime.now()}")

    def _create_run_metadata(self):
        meta = {'run_id': f'run_{self._CURRENT_RUN_TIMESTAMP}', 'start_time': datetime.now().isoformat(), 'config': {'device': self.DEVICE, 'num_buses': getattr(self, 'NUM_BUSES', 'auto')}}
        with open(os.path.join(self.CURRENT_RUN_DIR, 'run_metadata.json'), 'w') as f: json.dump(meta, f, indent=2)

    def finalize_run(self, summary=None):
        meta_file = os.path.join(self.CURRENT_RUN_DIR, 'run_metadata.json')
        if os.path.exists(meta_file):
            with open(meta_file) as f: meta = json.load(f)
            meta.update({'end_time': datetime.now().isoformat(), 'status': 'completed', 'results_summary': summary or {}})
            with open(meta_file, 'w') as f: json.dump(meta, f, indent=2)
        
        log_entry = {'run_id': f'run_{self._CURRENT_RUN_TIMESTAMP}', 'status': 'completed', **(summary or {})}
        log_path = os.path.join(self.EXPERIMENTAL_RESULTS_DIR, 'experiment_log.csv')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_exists = os.path.exists(log_path)
        with open(log_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=log_entry.keys())
            if not file_exists: w.writeheader()
            w.writerow(log_entry)
            
    def get_run_info(self): return {'run_id': f'run_{self._CURRENT_RUN_TIMESTAMP}', 'timestamp': self._CURRENT_RUN_TIMESTAMP, 'dir': self.CURRENT_RUN_DIR}
