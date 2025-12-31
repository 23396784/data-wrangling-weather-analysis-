"""
================================================================================
DATA WRANGLING & HEALTH DATA ANALYSIS: NHANES BODY MEASUREMENTS
================================================================================

A comprehensive data wrangling project analyzing body measurements from the
National Health and Nutrition Examination Survey (NHANES) dataset.

Skills Demonstrated:
- Data Loading & Multi-column Parsing
- Missing Value Handling
- Feature Engineering (BMI Calculation)
- Statistical Analysis & Comparison
- Correlation Analysis (Pearson & Spearman)
- Data Standardization (Z-Score)
- Visualization (Box Plots, Scatter Matrices)

Author: Victor Prefa
Institution: Deakin University
Course: MSc Data Science & Business Analytics

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple, Optional

# =============================================================================
# CONSTANTS
# =============================================================================

# Column indices for NHANES body measurement data
COLUMNS = {
    'weight': 0,      # Body weight (kg)
    'height': 1,      # Standing height (cm)
    'arm_length': 2,  # Upper arm length (cm)
    'leg_length': 3,  # Upper leg length (cm)
    'arm_circ': 4,    # Arm circumference (cm)
    'hip_circ': 5,    # Hip circumference (cm)
    'waist_circ': 6   # Waist circumference (cm)
}

# BMI Categories (WHO Classification)
BMI_CATEGORIES = {
    'Underweight': (0, 18.5),
    'Normal': (18.5, 25.0),
    'Overweight': (25.0, 30.0),
    'Obese': (30.0, float('inf'))
}


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_nhanes_data(filepath: str) -> np.ndarray:
    """
    Load NHANES body measurement data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        NumPy array with body measurements
        
    Example:
        >>> female = load_nhanes_data('nhanes_adult_female_bmx_2020.csv')
        >>> print(f"Loaded {female.shape[0]} records")
    """
    data = np.genfromtxt(filepath,
                         delimiter=',',
                         skip_header=1)
    return data


def validate_data(data: np.ndarray, name: str = "Dataset") -> Dict:
    """
    Validate data integrity and report missing values.
    
    Args:
        data: NumPy array to validate
        name: Name of the dataset for reporting
        
    Returns:
        Dictionary with validation results
    """
    total_values = data.size
    nan_count = np.isnan(data).sum()
    nan_per_column = np.isnan(data).sum(axis=0)
    complete_rows = np.sum(~np.isnan(data).any(axis=1))
    
    validation = {
        'name': name,
        'shape': data.shape,
        'total_values': total_values,
        'nan_count': nan_count,
        'nan_percentage': (nan_count / total_values) * 100,
        'nan_per_column': nan_per_column,
        'complete_rows': complete_rows,
        'complete_percentage': (complete_rows / data.shape[0]) * 100
    }
    
    return validation


def print_validation_report(validation: Dict) -> None:
    """Print formatted validation report."""
    print(f"\n{'='*60}")
    print(f"  Data Validation Report: {validation['name']}")
    print(f"{'='*60}")
    print(f"  Shape: {validation['shape']}")
    print(f"  Total values: {validation['total_values']:,}")
    print(f"  Missing values (NaN): {validation['nan_count']:,} ({validation['nan_percentage']:.2f}%)")
    print(f"  Complete rows: {validation['complete_rows']:,} ({validation['complete_percentage']:.1f}%)")
    print(f"  NaN per column: {validation['nan_per_column']}")
    print(f"{'='*60}")


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================

def calculate_bmi(weight_kg: np.ndarray, height_cm: np.ndarray) -> np.ndarray:
    """
    Calculate Body Mass Index (BMI).
    
    Formula: BMI = weight (kg) / height (m)²
    
    Args:
        weight_kg: Array of weights in kilograms
        height_cm: Array of heights in centimeters
        
    Returns:
        Array of BMI values
        
    Example:
        >>> bmi = calculate_bmi(np.array([70, 80]), np.array([175, 180]))
        >>> print(bmi)  # [22.86, 24.69]
    """
    height_m = height_cm / 100  # Convert to meters
    bmi = weight_kg / (height_m ** 2)
    return bmi


def classify_bmi(bmi_values: np.ndarray) -> Dict[str, int]:
    """
    Classify BMI values into WHO categories.
    
    Args:
        bmi_values: Array of BMI values
        
    Returns:
        Dictionary with counts per category
    """
    # Remove NaN values
    bmi_clean = bmi_values[~np.isnan(bmi_values)]
    
    counts = {}
    for category, (lower, upper) in BMI_CATEGORIES.items():
        count = np.sum((bmi_clean >= lower) & (bmi_clean < upper))
        counts[category] = count
        
    return counts


def add_bmi_column(data: np.ndarray) -> np.ndarray:
    """
    Add BMI as a new column to the dataset.
    
    Args:
        data: Original NHANES data array
        
    Returns:
        Data array with BMI column appended
    """
    weight = data[:, COLUMNS['weight']]
    height = data[:, COLUMNS['height']]
    bmi = calculate_bmi(weight, height)
    return np.column_stack([data, bmi])


# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================

def compute_descriptive_statistics(data: np.ndarray, 
                                    variable_name: str = "Variable") -> Dict:
    """
    Compute comprehensive descriptive statistics.
    
    Args:
        data: NumPy array (1D)
        variable_name: Name of the variable
        
    Returns:
        Dictionary with all statistics
    """
    # Remove NaN values
    clean_data = data[~np.isnan(data)]
    
    return {
        'variable': variable_name,
        'count': len(clean_data),
        'mean': np.mean(clean_data),
        'median': np.median(clean_data),
        'std': np.std(clean_data),
        'min': np.min(clean_data),
        'max': np.max(clean_data),
        'range': np.max(clean_data) - np.min(clean_data),
        'q1': np.percentile(clean_data, 25),
        'q3': np.percentile(clean_data, 75),
        'iqr': np.percentile(clean_data, 75) - np.percentile(clean_data, 25),
        'skewness': stats.skew(clean_data),
        'kurtosis': stats.kurtosis(clean_data)
    }


def compare_groups(group1: np.ndarray, group2: np.ndarray,
                   name1: str = "Group 1", name2: str = "Group 2") -> None:
    """
    Compare two groups with side-by-side statistics.
    
    Args:
        group1: First group data
        group2: Second group data
        name1: Name of first group
        name2: Name of second group
    """
    stats1 = compute_descriptive_statistics(group1, name1)
    stats2 = compute_descriptive_statistics(group2, name2)
    
    print(f"\n{'':15} {name1:>12} {name2:>12}")
    print("-" * 40)
    print(f"{'Count':15} {stats1['count']:>12,} {stats2['count']:>12,}")
    print(f"{'Mean':15} {stats1['mean']:>12.2f} {stats2['mean']:>12.2f}")
    print(f"{'Median':15} {stats1['median']:>12.2f} {stats2['median']:>12.2f}")
    print(f"{'Std Dev':15} {stats1['std']:>12.2f} {stats2['std']:>12.2f}")
    print(f"{'Min':15} {stats1['min']:>12.2f} {stats2['min']:>12.2f}")
    print(f"{'Max':15} {stats1['max']:>12.2f} {stats2['max']:>12.2f}")
    print(f"{'IQR':15} {stats1['iqr']:>12.2f} {stats2['iqr']:>12.2f}")
    print(f"{'Skewness':15} {stats1['skewness']:>12.2f} {stats2['skewness']:>12.2f}")


# =============================================================================
# DATA STANDARDIZATION FUNCTIONS
# =============================================================================

def standardize_zscore(data: np.ndarray) -> np.ndarray:
    """
    Apply z-score standardization.
    
    Formula: z = (x - mean) / std
    
    Args:
        data: NumPy array to standardize
        
    Returns:
        Standardized array with mean=0 and std=1
    """
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    return (data - mean) / std


# =============================================================================
# CORRELATION ANALYSIS FUNCTIONS
# =============================================================================

def compute_correlation_matrix(data: np.ndarray, 
                                labels: List[str],
                                method: str = 'pearson') -> np.ndarray:
    """
    Compute correlation matrix.
    
    Args:
        data: NumPy array (2D)
        labels: Variable labels
        method: 'pearson' or 'spearman'
        
    Returns:
        Correlation matrix
    """
    # Remove rows with NaN
    clean_data = data[~np.isnan(data).any(axis=1)]
    
    if method == 'pearson':
        return np.corrcoef(clean_data.T)
    elif method == 'spearman':
        return stats.spearmanr(clean_data)[0]
    else:
        raise ValueError("Method must be 'pearson' or 'spearman'")


def print_correlation_matrix(corr_matrix: np.ndarray, 
                              labels: List[str],
                              title: str = "Correlation Matrix") -> None:
    """Print formatted correlation matrix."""
    print(f"\n{title}")
    print("-" * 60)
    
    # Header
    print(f"{'':12}", end="")
    for label in labels:
        print(f"{label:>10}", end="")
    print()
    
    # Rows
    for i, label in enumerate(labels):
        print(f"{label:12}", end="")
        for j in range(len(labels)):
            print(f"{corr_matrix[i,j]:>10.3f}", end="")
        print()


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_bmi_boxplot(bmi_male: np.ndarray, bmi_female: np.ndarray,
                        title: str = "BMI Distribution by Gender") -> None:
    """
    Create horizontal box plot comparing BMI by gender.
    
    Args:
        bmi_male: Male BMI values
        bmi_female: Female BMI values
        title: Plot title
    """
    # Clean data
    bmi_m = bmi_male[~np.isnan(bmi_male)]
    bmi_f = bmi_female[~np.isnan(bmi_female)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot([bmi_m, bmi_f], 
                     labels=['Male', 'Female'], 
                     vert=False,
                     patch_artist=True)
    
    # Style
    colors = ['lightblue', 'lightpink']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('BMI (kg/m²)')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add BMI category lines
    for threshold, label in [(18.5, 'Underweight'), (25, 'Normal'), (30, 'Overweight')]:
        ax.axvline(x=threshold, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def create_scatter_matrix(data: np.ndarray, labels: List[str],
                           title: str = "Scatter Plot Matrix") -> None:
    """
    Create scatter plot matrix for body measurements.
    
    Args:
        data: NumPy array with measurements
        labels: Variable labels
        title: Plot title
    """
    # Remove NaN rows
    clean_data = data[~np.isnan(data).any(axis=1)]
    n_vars = len(labels)
    
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(12, 12))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal - show variable name
                ax.text(0.5, 0.5, labels[i], ha='center', va='center', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Scatter plot
                ax.scatter(clean_data[:, j], clean_data[:, i], 
                          alpha=0.3, s=1, c='blue')
            
            # Labels on edges only
            if i == n_vars - 1:
                ax.set_xlabel(labels[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=8)
    
    plt.tight_layout()
    plt.show()


def create_correlation_heatmap(corr_matrix: np.ndarray, 
                                labels: List[str],
                                title: str = "Correlation Heatmap") -> None:
    """
    Create correlation heatmap.
    
    Args:
        corr_matrix: Correlation matrix
        labels: Variable labels
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    # Add correlation values
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Correlation Coefficient')
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_complete_analysis(female_file: str, male_file: str) -> None:
    """
    Run complete NHANES health data analysis.
    
    Args:
        female_file: Path to female data CSV
        male_file: Path to male data CSV
    """
    print("\n" + "=" * 70)
    print("    DATA WRANGLING & HEALTH DATA ANALYSIS: NHANES BODY MEASUREMENTS")
    print("=" * 70)
    
    # Step 1: Load Data
    print("\n📥 STEP 1: Loading Data...")
    try:
        female = load_nhanes_data(female_file)
        male = load_nhanes_data(male_file)
        print(f"   ✓ Female data: {female.shape}")
        print(f"   ✓ Male data: {male.shape}")
    except FileNotFoundError:
        print("   ✗ Data files not found. Using simulated data...")
        np.random.seed(42)
        female = np.random.randn(4222, 7) * 10 + 50
        male = np.random.randn(4082, 7) * 12 + 55
    
    # Step 2: Validate Data
    print("\n🔍 STEP 2: Validating Data...")
    val_female = validate_data(female, "Female Dataset")
    val_male = validate_data(male, "Male Dataset")
    print_validation_report(val_female)
    print_validation_report(val_male)
    
    # Step 3: Calculate BMI
    print("\n📐 STEP 3: Feature Engineering - BMI Calculation...")
    bmi_female = calculate_bmi(female[:, 0], female[:, 1])
    bmi_male = calculate_bmi(male[:, 0], male[:, 1])
    print(f"   ✓ Female BMI calculated: {np.sum(~np.isnan(bmi_female)):,} valid values")
    print(f"   ✓ Male BMI calculated: {np.sum(~np.isnan(bmi_male)):,} valid values")
    
    # Step 4: BMI Classification
    print("\n📊 STEP 4: BMI Classification...")
    print("\n   Female BMI Categories:")
    for cat, count in classify_bmi(bmi_female).items():
        print(f"      {cat}: {count:,}")
    
    print("\n   Male BMI Categories:")
    for cat, count in classify_bmi(bmi_male).items():
        print(f"      {cat}: {count:,}")
    
    # Step 5: Comparative Statistics
    print("\n📈 STEP 5: Comparative BMI Statistics...")
    compare_groups(bmi_female, bmi_male, "Female", "Male")
    
    # Step 6: Correlation Analysis (Males)
    print("\n🔬 STEP 6: Correlation Analysis (Males)...")
    
    # Prepare male data with BMI
    male_with_bmi = add_bmi_column(male)
    selected_cols = [1, 0, 6, 5, 7]  # Height, Weight, Waist, Hip, BMI
    labels = ['Height', 'Weight', 'Waist', 'Hip', 'BMI']
    
    male_selected = male_with_bmi[:, selected_cols]
    
    # Compute correlations
    pearson = compute_correlation_matrix(male_selected, labels, 'pearson')
    print_correlation_matrix(pearson, labels, "Pearson Correlation Matrix")
    
    print("\n✅ Analysis Complete!")
    print("=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    run_complete_analysis(
        'nhanes_adult_female_bmx_2020.csv',
        'nhanes_adult_male_bmx_2020.csv'
    )
