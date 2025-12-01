from train import run_pipeline 
import pandas as pd

def main():
    """
    Main execution function for the project.
    """
    # Execute the entire ML pipeline, capturing the model and results
    model, results = run_pipeline()
    
    # --- ADDED CODE: Print the results ---
    print("\n--- Model Evaluation Results ---")
    
    # Check if results is a dictionary and print it using pandas for a clean table format
    if isinstance(results, dict):
        # Transpose the DataFrame (T) to show metrics as rows and sets (train/test) as columns
        results_df = pd.DataFrame(results).T 
        print(results_df)
    else:
        print("Error: Could not retrieve evaluation results.")
    # ------------------------------------

    # You can add further logic here, like saving the model or results
    print("\nPipeline finished successfully.")
    
if __name__ == "__main__":
    main()