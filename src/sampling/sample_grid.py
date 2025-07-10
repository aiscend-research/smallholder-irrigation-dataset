import sys
import os
import pandas as pd

# Add the project root to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.utils import get_data_root, save_data

class SampleGenerator:
    """A class to manage sampling from a spatial grid while tracking unique IDs."""

    def __init__(self, grid_path, sample_group_name):
        """
        Initialize the sample generator.

        Parameters:
            grid_path (str): Path to the full grid CSV file.
            sample_group_name (str): Name of the sample group (used for organizing saved files). 
            
        Every sample group has (or will have) a file within it that tracks all points that have already been sampled so that they are not repeated in subsequent samples. 
        
        Samples are saved under data/sampling/samples/sample_group_name/ with the filename being the country name, ag_thresh, and the range of sampled points.
        """
        self.grid_path = grid_path
        self.grid = pd.read_csv(grid_path)  # Load the grid once into memory
        self.sample_group_name = sample_group_name
        self.samples_dir = os.path.join(get_data_root(), f"sampling/samples/{self.sample_group_name}")
        self.sampled_points_file = self.samples_dir + "/sampled_points.txt"
        self.sampled_points = self._get_sampled_points()  # Load the last used ID

        # Remove previously sampled points from the grid
        self.grid = self.grid[~self.grid['id'].isin(self.sampled_points)]

    def _get_sampled_points(self):
        """
        Load the set of previously sampled points from file.
        """
        if os.path.exists(self.sampled_points_file):
            with open(self.sampled_points_file, "r") as f:
                sampled_points = f.read().strip().split("\n")
                if len(set(sampled_points)) < len(sampled_points):
                    print("Warning: There are dublicate sampled points in the list of sampled points. This can cause issues with the numbering of the samples.")
                return set(sampled_points)
        return set()
    
    def _update_sampled_points(self, new_sampled_points):
        """Update the set of sampled points in the tracking file."""

        # Ensure the directory exists before writing
        os.makedirs(os.path.dirname(self.sampled_points_file), exist_ok=True)

        # Write the new sampled points to the file
        with open(self.sampled_points_file, "a") as f:
            f.write("\n".join(new_sampled_points) + "\n")

        # Update the set of sampled points in the class
        self.sampled_points = set(self.sampled_points).union(new_sampled_points)

    def sample(self, num_samples, country="All", ag_thresh=0.05):
        """
        Sample a grid of points from the dataset without replacement.

        Parameters:
            num_samples (int): Number of samples to take.
            country (str): Country name to filter by (default: "All").
            ag_thresh (float): Minimum agriculture proportion threshold.

        Returns:
            pd.DataFrame: A DataFrame containing the sampled points.
        """
        # Filter by agriculture threshold
        sampled_grid = self.grid[self.grid['agriculture'] > ag_thresh]

        # Filter by country
        if country != "All":
            sampled_grid = sampled_grid[sampled_grid['country'] == country]

        # Sample the grid without replacement
        samples = sampled_grid.sample(num_samples)

        # Format for Collect
        samples = samples[['id', 'latitude', 'longitude']].rename(columns={
            'id': 'id', 'latitude': 'YCoordinate', 'longitude': 'XCoordinate'
        })

        # Sort the samples by latitude, then longitude
        samples = samples.sort_values(by=['YCoordinate', 'XCoordinate'])

        # Add the list of chosen ids to the set of sampled points
        self._update_sampled_points(samples['id'].tolist())

        # Save the sampled data
        filename = f"sampling/samples/{self.sample_group_name}/{country}_{ag_thresh}_n_{len(self.sampled_points) - num_samples + 1}-{len(self.sampled_points)}.csv"
        description = f"{num_samples} sampled grid points from {country} in areas with at least {ag_thresh} agriculture. Total samples in this sample group to date: {len(self.sampled_points)}"
        save_data(samples, filename, description=description, file_format="csv")

        return samples

# Example Usage
if __name__ == '__main__':
    # Initialize the sample generator with the grid file
    grid_loc = get_data_root() + '/sampling/grid/combined/agriculture_grid.csv'
    sampler = SampleGenerator(grid_loc, "random_sample")

    # # Generate samples
    # samples = sampler.sample(50, country="Zambia", ag_thresh=0.05)
    # sampler.sample(25, country="Zambia", ag_thresh=0.05)
    # sampler.sample(50, country="All", ag_thresh=0.05)
    for i in range(19):
        sampler.sample(25, country="Zambia", ag_thresh=0.05)