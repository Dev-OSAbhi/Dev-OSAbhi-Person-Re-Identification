import pandas as pd
import time

def process():
    start = time.time()
    print("Reading csv files...")
    cross_df = pd.read_csv('rankings_cross.csv')
    rgb_df = pd.read_csv('rankings_rgb.csv')
    depth_df = pd.read_csv('rankings_depth.csv')

    # Ensure column names are standard
    cross_df.columns = cross_df.columns.str.strip()
    rgb_df.columns = rgb_df.columns.str.strip()
    depth_df.columns = depth_df.columns.str.strip()

    cols = list(cross_df.columns[:4])
    print(f"Merge columns: {cols}")

    print("Merging dataframes...")
    # Select only the needed columns to avoid huge memory usage
    rgb_sub = rgb_df[cols + ['distance']].rename(columns={'distance': 'dist_rgb'})
    depth_sub = depth_df[cols + ['distance']].rename(columns={'distance': 'dist_depth'})

    # Merge cross with rgb and depth
    merged_df = cross_df.merge(rgb_sub, on=cols, how='left')
    merged_df = merged_df.merge(depth_sub, on=cols, how='left')

    # Calculate average distance where both exist
    mask = merged_df['dist_rgb'].notnull() & merged_df['dist_depth'].notnull()
    
    # We remove the average logic and restore the old logic for the cross track
    # which simply retains the original 'distance' values from rankings_cross.csv
    # merged_df.loc[mask, 'distance'] = (merged_df.loc[mask, 'dist_rgb'] + merged_df.loc[mask, 'dist_depth']) / 2.0
    
    updated_count = mask.sum()
    print(f"Retained original distances for {updated_count} rows that had matching RGB and Depth entries.")

    # Drop the temporary columns
    merged_df.drop(columns=['dist_rgb', 'dist_depth'], inplace=True)

    # Sort cross_df and update ranks
    print("Re-ranking based on new distances...")
    merged_df.sort_values(by=[cols[0], cols[1], 'distance'], inplace=True)
    merged_df['rank'] = merged_df.groupby([cols[0], cols[1]]).cumcount() + 1
    
    output_file = 'rankings_cross_new.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"Saved new cross rankings to {output_file} in {time.time() - start:.2f} seconds.")

if __name__ == '__main__':
    process()
