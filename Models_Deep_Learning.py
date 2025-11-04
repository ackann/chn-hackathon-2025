from Data_Loader_and_Merger import *
from Data_Splitter import *
from sklearn.metrics import explained_variance_score

def main():

    # --- Execute the aggregated version (runs the 'if' blocks) ---
    final_df_agg = merge_data(aggregate=True)
    print("\nAggregated Data Head:")
    print(final_df_agg.head())

    X_train, X_test, y_train, y_test = data_split_by_p_factor(aggregate=True) 
    
    if X_train is not None:
        print("\nReady for Modeling:")
        print(f"Features (X_train) head:\n{X_train.head()}")
        print(f"Target (y_train) head:\n{y_train.head()}")


    #explained_variance_score(y_true, y_pred)

    

if __name__ == "__main__":
    main()