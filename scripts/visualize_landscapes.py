import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization.plot_mosoa import plot_3d_landscape, plot_categorical_landscapes_grid
from src.benchmarks.functions import BENCHMARKS
from src.visualization.plot_mosoa import plot_3d_landscape, plot_categorical_landscapes_grid
from src.benchmarks.functions import BENCHMARKS
from scripts.benchmark_math_hpo import mock_training_pipeline
import shutil

def main():
    out_dir = "reports/mosoa/landscapes"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Mathematical Landscapes (F1-F23) - Grouped into grids
    print("\nGenerating Categorized 3D Landscape Grids...")
    print("-" * 65)
    
    groups = [
        # Unimodal (7 fns) -> Split into 4 and 3
        (['F1', 'F2', 'F3', 'F4'], "Unimodal Landscapes (F1-F4)", "math_unimodal_F1_F4.png"),
        (['F5', 'F6', 'F7'], "Unimodal Landscapes (F5-F7)", "math_unimodal_F5_F7.png"),
        # Multimodal (6 fns) -> One grid
        (['F8', 'F9', 'F10', 'F11', 'F12', 'F13'], "Multimodal Landscapes (F8-F13)", "math_multimodal_F8_F13.png"),
        # Fixed-Dim (10 fns) -> Split into 5 and 5
        (['F14', 'F15', 'F16', 'F17', 'F18'], "Fixed-Dimension Landscapes (F14-F18)", "math_fixed_dim_F14_F18.png"),
        (['F19', 'F20', 'F21', 'F22', 'F23'], "Fixed-Dimension Landscapes (F19-F23)", "math_fixed_dim_F19_F23.png"),
    ]

    for fn_names, title, filename in tqdm(groups, desc=f"{'Grid Phase':<15}", ncols=100):
        fns = []
        bounds_list = []
        target_dims = []
        
        for name in fn_names:
            config = BENCHMARKS[name]
            fns.append(config['fn'])
            
            b = config['bounds']
            if isinstance(b[0], list):
                plot_b = (b[0][0], b[0][1])
            else:
                plot_b = (b[0], b[1])
            
            # Zoom logic for clarity
            if name == 'F5': plot_b = (-5, 5)
            if name == 'F8': plot_b = (-500, 500)
            
            bounds_list.append(plot_b)
            dim = config.get('dim')
            target_dims.append(dim if dim is not None else 30)

        plot_categorical_landscapes_grid(
            fns, fn_names, bounds_list, title, 
            os.path.join(out_dir, filename), 
            target_dims, resolution=50
        )

    # 2. Mathematical HPO Tuning Landscape (formerly Proxy)
    print("\nGenerating 3D Landscape for Mathematical HPO (Research-Grade Surface)...")
    def proxy_wrapper(vec):
        params = {'learning_rate': vec[0], 'gcn_hidden': vec[1], 'lstm_hidden': 32, 'gru_hidden': 32, 'dropout': 0.2, 'num_layers': 3}
        orig_random = np.random.normal
        np.random.normal = lambda *args: 0 
        res = mock_training_pipeline(params)
        np.random.normal = orig_random
        return res

    plot_3d_landscape(proxy_wrapper, (0, 0.1), title="Mathematical HPO Benchmark Landscape (Multimodal Surface)", 
                      output_path=os.path.join(out_dir, "math_hpo_landscape_3d.png"))
    
    print(f"\nAll categorized landscapes generated in {out_dir}")

if __name__ == "__main__":
    main()
