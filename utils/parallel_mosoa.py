"""
Parallel Multi-Objective Seagull Optimization Algorithm (MoSOA) implementation.
Optimized for concurrent hyperparameter evaluation in high-memory environments.
"""

import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Tuple, Dict, Any
import time
import threading


class ParallelMoSOA:
    """
    Parallel implementation of Multi-Objective Seagull Optimization Algorithm.
    Evaluates multiple seagulls (parameter sets) concurrently for faster optimization.
    """
    
    def __init__(self, num_seagulls: int, max_iterations: int, 
                 lower_bounds: List[float], upper_bounds: List[float], 
                 dimension: int, objective_function: Callable,
                 max_workers: int = 4, batch_size: int = None):
        """
        Initialize Parallel MoSOA optimizer.
        
        Args:
            num_seagulls: Number of seagulls in the population
            max_iterations: Maximum number of iterations
            lower_bounds: Lower bounds for each dimension
            upper_bounds: Upper bounds for each dimension
            dimension: Problem dimension
            objective_function: Function to minimize
            max_workers: Maximum number of parallel workers
            batch_size: Number of seagulls to evaluate in parallel (default: max_workers)
        """
        self.num_seagulls = num_seagulls
        self.max_iterations = max_iterations
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.dimension = dimension
        self.objective_function = objective_function
        self.max_workers = max_workers
        self.batch_size = batch_size or max_workers
        
        # Initialize population
        self.population = self._initialize_population()
        self.fitness = np.full(num_seagulls, float('inf'))
        self.best_position = None
        self.best_fitness = float('inf')
        
        # History tracking
        self.history = []
        self.iteration_details = []
        
        # Thread safety
        self.evaluation_lock = threading.Lock()
        self.completed_evaluations = 0
        
    def _initialize_population(self) -> np.ndarray:
        """Initialize seagull population randomly within bounds."""
        population = np.random.uniform(
            low=self.lower_bounds,
            high=self.upper_bounds,
            size=(self.num_seagulls, self.dimension)
        )
        return population
    
    def _evaluate_batch_parallel(self, positions: np.ndarray, indices: List[int]) -> List[Tuple[int, float]]:
        """
        Evaluate a batch of positions in parallel.
        
        Args:
            positions: Array of positions to evaluate
            indices: Corresponding indices in the population
            
        Returns:
            List of (index, fitness) tuples
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all evaluations
            future_to_index = {
                executor.submit(self._safe_objective_evaluation, pos): (idx, pos)
                for idx, pos in zip(indices, positions)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                idx, pos = future_to_index[future]
                try:
                    fitness = future.result()
                    results.append((idx, fitness))
                    
                    with self.evaluation_lock:
                        self.completed_evaluations += 1
                        if self.completed_evaluations % 5 == 0:  # Progress update every 5 evaluations
                            print(f"🔄 Completed {self.completed_evaluations} evaluations...")
                            
                except Exception as e:
                    logging.error(f"Evaluation failed for seagull {idx}: {e}")
                    results.append((idx, float('inf')))
        
        return results
    
    def _safe_objective_evaluation(self, position: np.ndarray) -> float:
        """Safely evaluate objective function with error handling."""
        try:
            # Ensure position is within bounds
            clipped_position = np.clip(position, self.lower_bounds, self.upper_bounds)
            return self.objective_function(clipped_position)
        except Exception as e:
            logging.error(f"Objective function evaluation failed: {e}")
            return float('inf')
    
    def _update_best_solution(self, fitness_values: np.ndarray):
        """Update the best solution found so far."""
        min_idx = np.argmin(fitness_values)
        if fitness_values[min_idx] < self.best_fitness:
            self.best_fitness = fitness_values[min_idx]
            self.best_position = self.population[min_idx].copy()
    
    def _migration_behavior(self, iteration: int) -> np.ndarray:
        """
        Implement seagull migration behavior.
        Seagulls move towards the best position with some randomness.
        """
        new_population = np.zeros_like(self.population)
        
        for i in range(self.num_seagulls):
            if self.best_position is not None:
                # Migration towards best position with random component
                r1 = np.random.random(self.dimension)
                r2 = np.random.random(self.dimension)
                
                # Calculate migration vector
                migration_vector = r1 * (self.best_position - self.population[i])
                
                # Add random exploration
                exploration = r2 * (self.upper_bounds - self.lower_bounds) * 0.1
                
                # Update position
                new_population[i] = self.population[i] + migration_vector + exploration
            else:
                # If no best position yet, use random walk
                new_population[i] = self.population[i] + np.random.normal(0, 0.1, self.dimension)
            
            # Ensure bounds
            new_population[i] = np.clip(new_population[i], self.lower_bounds, self.upper_bounds)
        
        return new_population
    
    def _attacking_behavior(self, iteration: int) -> np.ndarray:
        """
        Implement seagull attacking behavior.
        Seagulls perform local search around promising areas.
        """
        new_population = np.zeros_like(self.population)
        
        for i in range(self.num_seagulls):
            if self.best_position is not None:
                # Local search around best position
                r = np.random.random(self.dimension)
                
                # Attacking intensity decreases over iterations
                attack_intensity = 2 - 2 * iteration / self.max_iterations
                
                # Calculate attacking vector
                attacking_vector = attack_intensity * r * (self.best_position - self.population[i])
                
                # Update position
                new_population[i] = self.population[i] + attacking_vector
            else:
                # Random local search
                new_population[i] = self.population[i] + np.random.normal(0, 0.05, self.dimension)
            
            # Ensure bounds
            new_population[i] = np.clip(new_population[i], self.lower_bounds, self.upper_bounds)
        
        return new_population
    
    def optimize(self) -> Tuple[float, np.ndarray, List[float], List[Dict[str, Any]]]:
        """
        Run the parallel MoSOA optimization.
        
        Returns:
            Tuple of (best_fitness, best_position, history, iteration_details)
        """
        print(f"🚀 Starting Parallel MoSOA optimization...")
        print(f"👥 Population size: {self.num_seagulls}")
        print(f"🔄 Max iterations: {self.max_iterations}")
        print(f"⚡ Max workers: {self.max_workers}")
        print(f"📦 Batch size: {self.batch_size}")
        
        start_time = time.time()
        
        # Initial evaluation of all seagulls
        print("🔄 Initial population evaluation...")
        self.completed_evaluations = 0
        
        # Evaluate population in batches
        for batch_start in range(0, self.num_seagulls, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.num_seagulls)
            batch_indices = list(range(batch_start, batch_end))
            batch_positions = self.population[batch_start:batch_end]
            
            batch_results = self._evaluate_batch_parallel(batch_positions, batch_indices)
            
            # Update fitness values
            for idx, fitness in batch_results:
                self.fitness[idx] = fitness
        
        # Update best solution
        self._update_best_solution(self.fitness)
        self.history.append(self.best_fitness)
        
        print(f"✅ Initial evaluation completed. Best fitness: {self.best_fitness:.6f}")
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            iteration_start_time = time.time()
            print(f"\n🔄 Iteration {iteration + 1}/{self.max_iterations}")
            
            # Phase 1: Migration behavior
            migration_population = self._migration_behavior(iteration)
            
            # Phase 2: Attacking behavior  
            attacking_population = self._attacking_behavior(iteration)
            
            # Combine behaviors (50% migration, 50% attacking)
            new_population = np.zeros_like(self.population)
            for i in range(self.num_seagulls):
                if np.random.random() < 0.5:
                    new_population[i] = migration_population[i]
                else:
                    new_population[i] = attacking_population[i]
            
            # Evaluate new population in batches
            self.completed_evaluations = 0
            new_fitness = np.full(self.num_seagulls, float('inf'))
            
            for batch_start in range(0, self.num_seagulls, self.batch_size):
                batch_end = min(batch_start + self.batch_size, self.num_seagulls)
                batch_indices = list(range(batch_start, batch_end))
                batch_positions = new_population[batch_start:batch_end]
                
                batch_results = self._evaluate_batch_parallel(batch_positions, batch_indices)
                
                # Update fitness values
                for idx, fitness in batch_results:
                    new_fitness[idx] = fitness
            
            # Selection: Keep better solutions
            for i in range(self.num_seagulls):
                if new_fitness[i] < self.fitness[i]:
                    self.population[i] = new_population[i]
                    self.fitness[i] = new_fitness[i]
            
            # Update best solution
            self._update_best_solution(self.fitness)
            self.history.append(self.best_fitness)
            
            iteration_time = time.time() - iteration_start_time
            
            # Store iteration details
            iteration_detail = {
                'iteration': iteration + 1,
                'best_fitness': self.best_fitness,
                'mean_fitness': np.mean(self.fitness),
                'std_fitness': np.std(self.fitness),
                'time_seconds': iteration_time,
                'evaluations': self.num_seagulls
            }
            self.iteration_details.append(iteration_detail)
            
            print(f"✅ Iteration {iteration + 1} completed in {iteration_time:.2f}s")
            print(f"🎯 Best fitness: {self.best_fitness:.6f}")
            print(f"📊 Mean fitness: {np.mean(self.fitness):.6f}")
            
            # Early stopping if no improvement for several iterations
            if len(self.history) > 10:
                recent_improvement = self.history[-10] - self.history[-1]
                if recent_improvement < 1e-6:
                    print(f"🛑 Early stopping: No significant improvement in last 10 iterations")
                    break
        
        total_time = time.time() - start_time
        print(f"\n🏁 Parallel MoSOA optimization completed!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"🎯 Best fitness achieved: {self.best_fitness:.6f}")
        print(f"📊 Total evaluations: {sum(detail['evaluations'] for detail in self.iteration_details) + self.num_seagulls}")
        
        return self.best_fitness, self.best_position, self.history, self.iteration_details


def parallel_soa(num_seagulls: int, max_iterations: int, 
                lower_bounds: List[float], upper_bounds: List[float], 
                dimension: int, objective_function: Callable,
                max_workers: int = 4, batch_size: int = None) -> Tuple[float, np.ndarray, List[float], List[Dict[str, Any]]]:
    """
    Convenience function to run Parallel MoSOA optimization.
    
    Args:
        num_seagulls: Number of seagulls in the population
        max_iterations: Maximum number of iterations
        lower_bounds: Lower bounds for each dimension
        upper_bounds: Upper bounds for each dimension
        dimension: Problem dimension
        objective_function: Function to minimize
        max_workers: Maximum number of parallel workers
        batch_size: Number of seagulls to evaluate in parallel
        
    Returns:
        Tuple of (best_fitness, best_position, history, iteration_details)
    """
    optimizer = ParallelMoSOA(
        num_seagulls=num_seagulls,
        max_iterations=max_iterations,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        dimension=dimension,
        objective_function=objective_function,
        max_workers=max_workers,
        batch_size=batch_size
    )
    
    return optimizer.optimize()
