import matplotlib.pyplot as plt

def pulse(magnitude, when, arr):
	arr[when] += magnitude
	return arr 

N = 100
I  = [0]*N		# Input current.
I  = pulse(1, 20, I) 
I  = pulse(1, 30, I)  
I  = pulse(1, 32, I)  
I  = pulse(1, 35, I) 
I  = pulse(1, 70, I)    
dt = 10e-7		# Time steps.
Tm = 10e-6		# Time duration.
t  = list(range(N))	# Time axis.
t  = [x * dt for x in t]
TauM = 0.9		# Time constant.

Vm = [0]*N
for i in range(2, N):
	if I[i] == 0:
		Vm[i] = Vm[i-1]*TauM
	else:
		Vm[i] = I[i] + Vm[i-1]*TauM
	
# Plot the signal and its derivative.
plt.figure(figsize=(10, 6))
plt.plot(t, I, label='Input Current', marker='o', linestyle='-', color='blue')
plt.plot(t, Vm, label="Membrane Potential", marker='x', linestyle='--', color='red')

# Adding labels, legend, and title
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Input current")
# plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Optional: Add y=0 line for reference
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
