import numpy as np
from scipy.special import gammaln
import matplotlib.pyplot as plt
from tabulate import tabulate

def volume_Bd(d):
    """
    Compute the volume of the unit ball in d dimensions.
    """
    ln_vol_Bd = (d / 2) * np.log(np.pi) - gammaln(1 + d / 2)
    return np.exp(ln_vol_Bd)

def way_one(d, N):
    """
    Monte Carlo integration sampling within the cube [-0.5, 0.5]^d.
    Estimates the volume of the intersection between the unit ball and the hypercube.
    """
    # Generate N random points uniformly in the cube [-0.5, 0.5]^d
    X = np.random.uniform(-0.5, 0.5, size=(N, d))
    
    # Count points inside unit ball (r^2 <= 1)
    r2 = np.sum(X**2, axis=1)
    count_inside_Bd = np.sum(r2 <= 1.0)
    
    # Estimate probability and compute standard error
    p_hat = count_inside_Bd / N
    vol_cube = 1.0  # Volume of [-0.5, 0.5]^d is 1
    est_vol = p_hat * vol_cube
    SE_est_vol = np.sqrt(p_hat * (1 - p_hat) / N) * vol_cube
    return est_vol, SE_est_vol

def way_two(d, N):
    """
    Monte Carlo integration sampling within the unit ball using inverse transform.
    Estimates the volume of the intersection between the unit ball and the hypercube.
    """
    # Generate random directions uniformly on unit sphere
    Z = np.random.normal(0, 1, size=(N, d))
    Z_norm = np.linalg.norm(Z, axis=1, keepdims=True)
    Z_unit = Z / Z_norm
    
    # Generate radii using inverse transform method for uniform distribution in ball
    U = np.random.uniform(0, 1, size=(N, 1))
    r = U ** (1 / d)
    
    # Scale unit vectors by radii
    X = Z_unit * r
    
    # Check which points lie in the cube [-0.5, 0.5]^d
    in_cube = np.all(np.abs(X) <= 0.5, axis=1)
    fraction_in_cube = np.mean(in_cube)
    SE_fraction_in_cube = np.sqrt(fraction_in_cube * (1 - fraction_in_cube) / N)
    
    # True volume of unit ball
    vol_Bd = volume_Bd(d)
    
    # Estimated volume of intersection
    est_vol = fraction_in_cube * vol_Bd
    SE_est_vol = vol_Bd * SE_fraction_in_cube
    
    return est_vol, SE_est_vol, vol_Bd, fraction_in_cube

def run_simulation(dims=[5, 10, 15, 20], N=int(1e7)):
    """
    Run Monte Carlo simulations for multiple dimensions.
    """
    result_table = []
    
    for d in dims:
        print(f"Running simulations for dimension d = {d} with N = {N} samples...")
        
        est_vol_way_one, SE_way_one = way_one(d, N)
        est_vol_way_two, SE_way_two, vol_Bd, fraction_in_cube = way_two(d, N)
        
        result = {
            'Dimension d': d,
            'Way One Volume': est_vol_way_one,
            'Way One SE': SE_way_one,
            'Way Two Volume': est_vol_way_two,
            'Way Two SE': SE_way_two,
            'True Ball Volume': vol_Bd,
            'Fraction in Cube': fraction_in_cube
        }
        result_table.append(result)
        
    print_results(result_table)
    plot_results(result_table)
    return result_table

def print_results(results):
    """
    Print results in a formatted table with standard errors.
    """
    headers = ['d', 'Way One Vol.', 'Way One SE', 'Way Two Vol.', 'Way Two SE', 'True Ball Vol.', 'Fraction in Cube']
    rows = [[
        r['Dimension d'],
        f"{r['Way One Volume']:.6e}",
        f"{r['Way One SE']:.6e}",
        f"{r['Way Two Volume']:.6e}",
        f"{r['Way Two SE']:.6e}",
        f"{r['True Ball Volume']:.6e}",
        f"{r['Fraction in Cube']:.6e}"
    ] for r in results]
    
    print("Simulation Results:")
    print(tabulate(rows, headers=headers, tablefmt='grid'))

def plot_results(results):
    """
    Plot volume estimates and true volumes with error bars and linear y-scale.
    """
    dims = [r['Dimension d'] for r in results]
    way_one_vols = [r['Way One Volume'] for r in results]
    way_two_vols = [r['Way Two Volume'] for r in results]
    true_vols = [r['True Ball Volume'] for r in results]
    way_one_SEs = [r['Way One SE'] for r in results]
    way_two_SEs = [r['Way Two SE'] for r in results]

    plt.figure(figsize=(10, 6))
    
    # Way One: Large diamonds with dashed lines and error bars
    plt.errorbar(dims, way_one_vols, yerr=way_one_SEs, fmt='D--', color='red', 
                 label='Way One (Cube Sampling)', 
                 markersize=10, linewidth=2, capsize=5)
    
    # Way Two: Stars with dotted lines and error bars
    plt.errorbar(dims, way_two_vols, yerr=way_two_SEs, fmt='*:', color='blue', 
                 label='Way Two (Ball Sampling)', 
                 markersize=12, linewidth=2, capsize=5)
    
    # True volume: Triangles with solid lines
    plt.plot(dims, true_vols, '^-', color='black', 
             label='True Ball Volume', 
             markersize=8, linewidth=2)
    
    plt.xlabel('Dimension $d$')
    plt.ylabel('Volume ')
    plt.title('Volume Estimates vs Dimension')
    plt.grid(True)
    plt.legend(fontsize=10)
    # plt.yscale('log') 
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)  
    run_simulation()
