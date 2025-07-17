# Contributing to the Smallholder Dry Season Irrigation Dataset

Thank you for your interest in contributing to this project! This document outlines the key guidelines for managing data and metadata to ensure consistency and reproducibility.

## General Contribution Notes
- Commit frequently with detailed commit messages and descriptions, especially when actively collaborating with others. 
- Update README.md files obsessively as changes to the repository and sub-repositories happen. 

## Data and Metadata Storage Guidelines

### **1. Data Storage:**
- **Directory Structure:**
  - Store data in relevant folders under `data/` that correspond with the `src` module they were created under (e.g., `data/sampling/`, `data/labels/`, `data/features/`).
  - Use descriptive file names that clearly indicate the content and version if applicable.
- **Cluster Data Location:**
  - On the UCSB GRIT ERI servers, all shared data should be stored in `/home/waves/data/smallholder-irrigation-dataset/data/`. The utility function `get_data_root()` will automatically detect and use this location when running on the cluster.
  - **Note:** The cluster data folder is not synchronized with the `data/` folder in the GitHub repository. If you add data to the cluster that should also be available in the GitHub repo (e.g., small files), you must manually add it to the repo's `data/` folder. Likewise, if you push data to the GitHub repo, you must manually add it to the cluster data folder.

### **2. Data Saving Protocol:**
- Use the `save_data()` utility function to ensure data and metadata are saved consistently. This function will automatically create a `.json` metadata file for each dataset.
- **Metadata and Documentation:**
  - Every data file in the shared cluster location or saved to the repository should have either:
    - An associated `.json` metadata file (created by `save_data()`), or
    - A `.README` file in the same directory explaining the contents and purpose of the files.
  - If you are generating a large number of very similar files, a single README explaining the batch is sufficient.
- **Delete Unused Files:**Delete files that are no longer used or have been moved elsewhere, but always retain raw data for reproducibility.

### **3. Configuration Management:**
- Paths and environment-specific variables can be stored in `config.yaml`.
- Never hardcode file paths within scripts; use helper functions like `get_data_root()`.

### **4. Data in the .gitignore:**
- By default, data should be listed in the `.gitignore`. Only add data files to the repository if they are small and useful for others.

Thank you for maintaining the quality and consistency of this project!
