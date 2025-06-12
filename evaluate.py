import os
import json
import sys
import argparse
from tqdm import tqdm
from collections import defaultdict
from model_scripts import *

# Argument parser setup
parser = argparse.ArgumentParser(description="Evaluate actions based on predicted coordinates.")
parser.add_argument('--model_to_run', type=str, required=False, default='os_atlas_baseline', help='Model to run for coordinate prediction')
args = parser.parse_args()

# Default path setup
# sys.path.insert(0, "your path to the repo directory")

def get_history(current_step, max_steps, path_to_history):

    formatted_history = ""
    
    try:
        with open(path_to_history, "r") as history_file:
            history_data = json.load(history_file)
        
        start_idx = max(0, current_step - max_steps)
        
        relevant_history = history_data[start_idx:current_step]
        
        history_items = []
        for i, hist_item in enumerate(relevant_history):
            step_num = start_idx + i + 1
            instruction = hist_item.get("instruction", "")
            action = hist_item.get("extracted_action", "")
            history_items.append(f"{i+1}. Instruction: {instruction}\n   Action: {action}")
        
        formatted_history = "\n".join(history_items)
        
    except FileNotFoundError:
        print(f"Warning: History file not found at {path_to_history}")
    except Exception as e:
        print(f"Error processing history: {e}")
    
    return formatted_history


def evaluate_actions(base_dir, episodes=None, output_dir=None, history_steps=None, image_output_dir=None, model_to_run="os_atlas"):
    if output_dir is None:
        output_dir = "output_files/evaluation_results/"
    
    if model_to_run == "os_atlas":
        from model_scripts import os_atlas_baseline
        func = os_atlas_baseline.get_coordinate
    elif model_to_run == "os_atlas_gpt":
        from model_scripts import os_atlas_baseline_gpt
        func = os_atlas_baseline_gpt.get_coordinate
    elif model_to_run == "uground":
        from model_scripts import uground_baseline
        func = uground_baseline.get_coordinate
    elif model_to_run == "uground_gpt":
        from model_scripts import uground_baseline_gpt
        func = uground_baseline_gpt.get_coordinate
    elif model_to_run == "aria_ui":
        from model_scripts import aria_ui_baseline
        func = aria_ui_baseline_gpt.get_coordinate
    elif model_to_run == "aria_ui_gpt":
        from model_scripts import aria_ui_baseline_gpt
        func = aria_ui_baseline_gpt.get_coordinate
    elif model_to_run == "set_of_mark":
        from model_scripts import set_of_mark_baseline
        func = set_of_mark_baseline.get_coordinate
    elif model_to_run == "o1":
        from model_scripts import uground_baseline_o1
        func = uground_baseline_o1.get_coordinate
    elif model_to_run == "claude":
        from model_scripts import uground_baseline_claude
        func = uground_baseline_claude.get_coordinate
    elif model_to_run == 'gpt_reasoning':
        from model_scripts import gpt_reasoning
        func = gpt_reasoning.get_coordinate
    elif model_to_run == 'o1_reasoning':
        from model_scripts import o1_reasoning
        func = o1_reasoning.get_coordinate
    elif model_to_run == 'claude_reasoning':
        from model_scripts import claude_reasoning
        func = claude_reasoning.get_coordinate
    os.makedirs(output_dir, exist_ok=True)
    
    # If no specific episodes provided, find all available episodes
    if episodes is None:
        episodes = []
        for item in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, item)) and item.isdigit():
                episodes.append(int(item))
    
    episodes.sort()
    print(f"Evaluating episodes: {episodes}")
    
    total = 0
    correct = 0
    skipped = 0
    results_by_episode = {}
    total_valid_count = 0
    total_distance = 0
    
    # Variables for task tracking
    all_tasks = defaultdict(list)  # Maps task_id -> list of success/failure for steps
    task_by_episode = defaultdict(lambda: defaultdict(list))  # Maps episode -> task_id -> list of success/failure
    
    for episode in tqdm(episodes, desc="Processing episodes"):
        episode_str = str(episode)
        episode_path = os.path.join(base_dir, episode_str)
        questions_path = os.path.join(episode_path, "questions_with_task_updated.json")
        history_path = os.path.join(episode_path, "extracted_actions_GT.json")
        
        episode_total = 0
        episode_correct = 0
        episode_skipped = 0
        episode_total_distance = 0
        episode_valid_count = 0
        
        try:
            with open(questions_path, "r") as json_file:
                questions_data = json.load(json_file)

            print(f"\nEvaluating episode {episode}:")
            # Process each item in the questions data
            for i, item in tqdm(enumerate(questions_data), desc="Evaluating episode"):
                # Get task ID from the item
                task_id = item.get("task", f"unknown_{i}")
                
                # Skip items where bounding_box is null, indicates it's not a click action
                if item["bounding_box"] is None:
                    episode_skipped += 1
                    skipped += 1
                    # For non-click actions, we don't count them in task success/progress
                    continue

                config_data = {
                    "bounding_box": item["bounding_box"],
                    "final_image": os.path.join(episode_path, item["final_image"]),
                    "instruction": item["instruction"],
                    "action": item["action"]
                }
                
                # Get formatted history string if history_steps is specified
                history_string = ""
                if history_steps is not None and os.path.exists(history_path):
                    history_string = get_history(i, history_steps, history_path)
                
                # Call your coordinate prediction function with history_string
                x, y = func(config_data, history_string, base_dir, image_output_dir)
                
                success = False
                if x is None and y is None:
                    print(f"  Item {i+1}: Invalid Output")
                else:
                    match = False
                    episode_valid_count += 1
                    total_valid_count += 1
                    # Check if bounding_box is a list or a single dict
                    bbox_data = item["bounding_box"]
                    if not isinstance(bbox_data, list):
                        bbox_data = [bbox_data]
                    min_distance = 100000000
                    for bbox in bbox_data:
                        distance = 0
                        x_min = bbox["x"]
                        x_max = bbox["x"] + bbox["width"]
                        y_min = bbox["y"]
                        y_max = bbox["y"] + bbox["height"]
                        if x < x_min:
                            distance += (x_min - x)
                        elif x > x_max:
                            distance += (x - x_max)
                        if y < y_min:
                            distance += (y_min - y)
                        elif y > y_max:
                            distance += (y - y_max)
                        if distance < min_distance:
                            min_distance = distance
                        if bbox["x"] <= x <= bbox["x"] + bbox["width"] and bbox["y"] <= y <= bbox["y"] + bbox["height"]:
                            match = True

                    episode_total_distance += min_distance
                    total_distance += min_distance
                            
                    if match:
                        print(f"  Item {i+1}: ✓ Correct match found")
                        episode_correct += 1
                        correct += 1
                        success = True
                    else:
                        print(f"  Item {i+1}: ✗ No match bounding box found")
                
                # Record success/failure for task progress tracking
                all_tasks[task_id].append(success)
                task_by_episode[episode][task_id].append(success)
                        
                episode_total += 1
                total += 1
            
            # Calculate episode-specific task metrics
            episode_progress = calculate_average_progress(task_by_episode[episode])
            episode_success_rate = calculate_task_success_rate(task_by_episode[episode])
                    
            # Store results for this episode
            if episode_total > 0:
                episode_accuracy = episode_correct / episode_total * 100
            else:
                episode_accuracy = 0
            
            if episode_valid_count > 0:
                episode_average_distance = episode_total_distance / episode_valid_count
            else:
                episode_average_distance = 0
                
            results_by_episode[episode] = {
                "total": episode_total,
                "correct": episode_correct,
                "skipped": episode_skipped,
                "accuracy": episode_accuracy,
                "average_distance": episode_average_distance,
                "average_progress": episode_progress,
                "task_success_rate": episode_success_rate
            }
            
            print(f"Episode {episode} accuracy: {episode_correct}/{episode_total} ({episode_accuracy:.2f}%)")
            print(f"Episode {episode} average distance: {episode_average_distance}")
            print(f"Episode {episode} average progress: {episode_progress:.2f}%")
            print(f"Episode {episode} task success rate: {episode_success_rate:.2f}%")
            print(f"Skipped {episode_skipped} items with null bounding boxes")
            
        except Exception as e:
            print(f"Error processing episode {episode}: {e}")
    
    # Calculate overall task metrics
    overall_progress = calculate_average_progress(all_tasks)
    overall_success_rate = calculate_task_success_rate(all_tasks)
    
    # Calculate and display overall results
    if total > 0:
        overall_accuracy = correct / total * 100
    else:
        overall_accuracy = 0

    if total_valid_count > 0:
        overall_average_distance = total_distance / total_valid_count
    else:
        overall_average_distance = 0
        
    print("\n--- OVERALL RESULTS ---")
    print(f"Accuracy: {correct}/{total} ({overall_accuracy:.2f}%)")
    print(f"Average Distance: {overall_average_distance}")
    print(f"Average Progress: {overall_progress:.2f}%")
    print(f"Task Success Rate: {overall_success_rate:.2f}%")
    print(f"Skipped {skipped} items as they are non-click actions")
    
    # Save results to JSON file
    results = {
        "overall": {
            "total": total,
            "correct": correct,
            "skipped": skipped,
            "accuracy": overall_accuracy,
            "average_distance": overall_average_distance,
            "average_progress": overall_progress,
            "task_success_rate": overall_success_rate
        },
        "episodes": results_by_episode
    }
    
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {results_path}")


def calculate_average_progress(tasks_dict):
    """
    Calculate average progress across all tasks
    
    Progress is defined as the percentage of consecutive successful 
    steps before the first failure relative to the total steps in the task.
    
    Args:
        tasks_dict: Dictionary mapping task_id -> list of True/False values
    
    Returns:
        float: Average progress percentage
    """
    if not tasks_dict:
        return 0.0
    
    total_progress = 0.0
    num_tasks = len(tasks_dict)
    
    for task_id, results in tasks_dict.items():
        if not results:
            continue
            
        total_steps = len(results)
        # Count consecutive successes from the beginning
        consecutive_successes = 0
        for success in results:
            if success:
                consecutive_successes += 1
            else:
                break
                
        # Calculate progress for this task
        task_progress = (consecutive_successes / total_steps) * 100 if total_steps > 0 else 0
        total_progress += task_progress
    
    # Average across all tasks
    return total_progress / num_tasks if num_tasks > 0 else 0.0


def calculate_task_success_rate(tasks_dict):
    """
    Calculate task success rate
    
    A task is successful if all steps were completed successfully.
    
    Args:
        tasks_dict: Dictionary mapping task_id -> list of True/False values
    
    Returns:
        float: Task success rate percentage
    """
    if not tasks_dict:
        return 0.0
    
    successful_tasks = 0
    num_tasks = len(tasks_dict)
    
    for task_id, results in tasks_dict.items():
        if not results:
            continue
            
        # A task is successful if all steps are True
        if all(results):
            successful_tasks += 1
    
    # Calculate success rate
    return (successful_tasks / num_tasks) * 100 if num_tasks > 0 else 0.0


if __name__ == "__main__":
    '''
    Args:
        base_dir: where the dataset is
        episodes: episodes to evaluate, None to evaluate all
        output_dir: where to output evaluation results
        history_steps: number of history steps to give, None for no history
    '''

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    base_dir = os.path.join(script_dir, "full_human_dataset")
    episodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    output_dir = os.path.join(script_dir, "output_files", "evaluation_results")
    history_steps = 10
    image_output_dir = os.path.join(script_dir, "output_files", "output_images")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    
    print(f"Model selected: {args.model_to_run}")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    if not os.path.exists(base_dir):
        print(f"Dataset directory not found: {base_dir}")
        print("Please ensure your dataset is in the 'data/full_human_dataset' directory")
        print("Or modify the base_dir variable in the script to point to your dataset location")
    
    evaluate_actions(base_dir, episodes, output_dir, history_steps, image_output_dir, args.model_to_run)
