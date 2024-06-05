# # Irem Keskin final project AA222

#selected airfoils in order
from math import atan
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

file_info = {
    'DU21': '/Users/iremkeskin/Desktop/AA222/finalproject/DU21_A17.dat',
    'DU25': '/Users/iremkeskin/Desktop/AA222/finalproject/DU25_A17.dat',
    'DU30': '/Users/iremkeskin/Desktop/AA222/finalproject/DU30_A17.dat',
    'DU35': '/Users/iremkeskin/Desktop/AA222/finalproject/DU35_A17.dat',
    'DU40': '/Users/iremkeskin/Desktop/AA222/finalproject/DU40_A17.dat',
    'NACA64': '/Users/iremkeskin/Desktop/AA222/finalproject/NACA64_A17.dat'
}

def read_and_clean_data(file_path, expected_columns=4):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into columns
            columns = line.strip().split()
            # Check if the line has the expected number of columns
            if len(columns) == expected_columns:
                # Convert columns to float
                columns = [float(col) for col in columns]
                # Check if the first column is between 0 and 18
                if 0 <= columns[0] <= 18:
                    data.append(columns)
    return np.array(data)


def objective_function(alpha, alpha_range, Cl, Cd):
    drag_to_lift = np.interp(alpha, alpha_range, Cd / Cl)
    return drag_to_lift

# Initialize a dictionary to store the cleaned data and the new optimal data
cleaned_data = {}
optimal_data = []

for name, file_path in file_info.items():
    try:
        # Read and clean the data
        data = read_and_clean_data(file_path)
        
        # Check if data is empty
        if len(data) == 0:
            print(f"Error: No valid data in file {file_path}")
            continue
        
        cleaned_data[name] = data
        
        # Extract columns for plotting
        col1 = np.array([row[0] for row in data])
        col2 = np.array([row[1] for row in data])
        col3 = np.array([row[2] for row in data])
        
        col_ratio = np.divide(col3, col2, out=np.zeros_like(col3), where=col2 != 0)
        data = np.column_stack((data, col_ratio))
        
        # Perform Powell's method to find optimal alpha
        initial_alpha = col1[np.argmax(col_ratio)]
        result = minimize(objective_function, initial_alpha, args=(col1, col2, col3), method='Powell', bounds=[(min(col1), max(col1))])
        optimal_alpha = result.x[0]
        
        # Find corresponding Cl and Cd for optimal alpha
        optimal_Cl = np.interp(optimal_alpha, col1, col2)
        optimal_Cd = np.interp(optimal_alpha, col1, col3)
        
        optimal_data.append([optimal_alpha, optimal_Cl, optimal_Cd])
        print(f'Optimal alpha for {name}: {optimal_alpha}')
        
        # Plot Column 2 and Column 3 vs Column 1
        plt.figure(figsize=(10, 5))
        plt.plot(col1, col2, marker='o', linestyle='-', label='Coeff. of Lift')
        plt.plot(col1, col3, marker='x', linestyle='-', label='Coeff. of Drag')
        plt.axvline(optimal_alpha, color='r', linestyle='--', label='Optimal Alpha')
        plt.xlabel('Angle of attack [deg]')
        plt.ylabel('Values')
        plt.title(f'{name}: Coefficient of Lift and Drag vs Angle of Attack')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except IOError:
        print(f"Error: An error occurred while reading the file {file_path}.")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")

optimal_data = np.array(optimal_data)
np.savetxt('/Users/iremkeskin/Desktop/AA222/finalproject/optimal_data.csv', optimal_data, delimiter=',', header='Optimal Alpha,Cl,Cd', comments='')

print("Optimal data saved to 'optimal_data.csv':")
print(optimal_data)

#################################################################################################

# Constants and parameters
blades = 3
tip_speed_ratio = 8
wind_speed = 10.4  # m/s
radius = 120  # meters
air_density = 1.225  # kg/m^3
hubrad = 0.25 * radius
length = radius - hubrad  # blade length without the hub

# Optimization table data
# First column: angle of attack, second column: coefficient of lift
design = np.array([
    [0, 0], [0, 0],  # cyl
    [6.0000, 0.9670], [6.0000, 0.9670], # 40
    [8.0000, 1.2600], [8.0000, 1.2600],  # 35
    [7.5000, 1.2560], # 30
    [5.0000, 1.0620], [5.0000, 1.0620],  [5.0000, 1.0620], # 25
    [3.5000, 0.9480], [3.5000, 0.9480],  # 21
    [5.0000, 1.0110], [5.0000, 1.0110], [5.0000, 1.0110], [5.0000, 1.0110], [5.0000, 1.0110]  # 64
])

# Max thickness of each foil
thic = np.array([1, 1, .4, .4, .35, .35, .3, .25, .25, .25, .21, .21, .18, .18, .18, .18, .18])

# Calculations
start, step, stop = 2.647058824, 5.294117647, 90 #from middle of the first foil to length of 90 by pitch e 
nodes = np.arange(start, stop, step)
nodes_normalized = nodes / length
tip_speed_i = tip_speed_ratio * nodes_normalized
phi = (2/3) * np.arctan(1. / tip_speed_i)

# Chord calculation using Equation 4
with np.errstate(divide='ignore', invalid='ignore'):
    chord = np.where(design[:, 1] == 0, 4, (16 * np.pi * length**2) / (9 * blades * design[:, 1] * tip_speed_ratio**2 * nodes))

pitch = np.rad2deg(phi) - design[:, 0]
twist = np.clip(pitch, None, 20)  # Ensure twist values do not exceed 0 degrees
thicc = thic * chord
beta = phi - np.deg2rad(twist)
alpha = np.deg2rad(pitch)

Cl = design[:, 1]
Cd = 0.01 * np.ones_like(Cl) 
lift = 0.5 * air_density * (tip_speed_i * wind_speed) ** 2 * chord * Cl
drag = 0.5 * air_density * (tip_speed_i * wind_speed) ** 2 * chord * Cd

sigma = (blades * chord) / (2 * np.pi * nodes)
a = 1 / (4 * np.sin(phi) ** 2 / (sigma * Cl))
a = np.clip(a, 0, 0.5)


Cp_local = (16/27) * a * (1 - a)**2 * tip_speed_ratio
Cp_avg = np.mean(Cp_local)
print(f"Average Power Coefficient (Cp): {Cp_avg}")

# Plot chord, twist, and thickness vs. r_i/R
plt.figure(figsize=(10, 5))
plt.plot(nodes_normalized, chord, marker='o', linestyle='-', color='b')
plt.xlabel('Nodes / Length')
plt.ylabel('Chord (meters)')
plt.title('Chord vs r_i/R')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(nodes_normalized, twist, marker='x', linestyle='-', color='r')
plt.xlabel('Nodes / Length')
plt.ylabel('Twist (degrees)')
plt.title('Twist vs r_i/R')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(nodes_normalized, thicc, marker='s', linestyle='-', color='g')
plt.xlabel('Nodes / Length')
plt.ylabel('Thickness (meters)')
plt.title('Thickness vs r_i/R')
plt.grid(True)
plt.show()

swept_area = np.pi * length**2
P = 0.5 * Cp_avg * air_density * swept_area * wind_speed**3
print(f"Power Output (P): {P} Watts")
