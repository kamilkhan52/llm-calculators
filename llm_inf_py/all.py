import json
import os
import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
import math

def calculate_memory_requirements(input_data):
    nl = input_data['num_layers']
    hd = input_data['hidden_dim']
    bsz = input_data['batch_size']
    seq = input_data['seq_length']
    ci = input_data['checkpoint_interval']
    attn_heads = input_data['attn_heads']

    # Model States
    model_states_memory = 240 * nl * hd ** 2

    # Activation Memory (without checkpointing)
    activation_memory = 2 * bsz * seq * hd * nl

    # Activation Checkpoints
    activation_checkpoints_memory = 2 * bsz * seq * hd * nl / ci

    # Model State Working Memory (MSWM)
    mswm = 16 * hd ** 2

    # Activation Working Memory (AWM)
    awm = bsz * seq * ci * (16 * hd + 2 * attn_heads * seq)

    # Convert memory to specified unit (default: GB)
    unit = input_data.get('unit', 'GB')
    if unit == 'TB':
        divisor = 1024 ** 4
    elif unit == 'GB':
        divisor = 1024 ** 3
    elif unit == 'MB':
        divisor = 1024 ** 2
    elif unit == 'KB':
        divisor = 1024
    else:
        print(f"Error: Unit {unit} not available.")
        exit(-1)

    return {
        'model_states_memory': model_states_memory / divisor,
        'activation_memory': activation_memory / divisor,
        'activation_checkpoints_memory': activation_checkpoints_memory / divisor,
        'mswm': mswm / divisor,
        'awm': awm / divisor,
        'unit': unit
    }

def update_memory_requirements(event=None):
    input_data = {
        'num_layers': int(num_layers_entry.get()),
        'hidden_dim': int(hidden_dim_entry.get()),
        'batch_size': int(batch_size_entry.get()),
        'seq_length': int(seq_length_entry.get()),
        'checkpoint_interval': int(checkpoint_interval_entry.get()),
        'attn_heads': int(attn_heads_entry.get()),
        'unit': unit_var.get(),
        'gpu_memory_gb': float(gpu_memory_entry.get())
    }
    memory_requirements = calculate_memory_requirements(input_data)

    model_states_var.set(f"{memory_requirements['model_states_memory']:.2f} {memory_requirements['unit']}")
    activation_var.set(f"{memory_requirements['activation_memory']:.2f} {memory_requirements['unit']}")
    activation_checkpoints_var.set(f"{memory_requirements['activation_checkpoints_memory']:.2f} {memory_requirements['unit']}")
    mswm_var.set(f"{memory_requirements['mswm']:.2f} {memory_requirements['unit']}")
    awm_var.set(f"{memory_requirements['awm']:.2f} {memory_requirements['unit']}")

    total_memory_required = memory_requirements['model_states_memory'] + memory_requirements['activation_checkpoints_memory']

    if memory_requirements['unit'] == 'TB':
        total_memory_required *= 1024
    elif memory_requirements['unit'] == 'GB':
        pass
    elif memory_requirements['unit'] == 'MB':
        total_memory_required /= 1024
    else:  # Assuming bytes
        total_memory_required /= (1024 * 1024 * 1024)

    num_gpus_required = math.ceil(total_memory_required / input_data['gpu_memory_gb'])
    num_gpus_var.set(str(num_gpus_required))

def load_input_file():
    selected_file = file_var.get()
    input_file = os.path.join("models", selected_file)

    with open(input_file, 'r') as file:
        input_data = json.load(file)

    num_layers_entry.delete(0, tk.END)
    num_layers_entry.insert(0, input_data['num_layers'])
    hidden_dim_entry.delete(0, tk.END)
    hidden_dim_entry.insert(0, input_data['hidden_dim'])
    batch_size_entry.delete(0, tk.END)
    batch_size_entry.insert(0, input_data['batch_size'])
    seq_length_entry.delete(0, tk.END)
    seq_length_entry.insert(0, input_data['seq_length'])
    checkpoint_interval_entry.delete(0, tk.END)
    checkpoint_interval_entry.insert(0, input_data['checkpoint_interval'])
    attn_heads_entry.delete(0, tk.END)
    attn_heads_entry.insert(0, input_data['attn_heads'])
    unit_var.set(input_data.get('unit', 'GB'))
    gpu_memory_entry.delete(0, tk.END)
    gpu_memory_entry.insert(0, input_data.get('gpu_memory_gb', 80))

    update_memory_requirements()

def calculate_bandwidth_requirements(input_data):
    nl = input_data['num_layers']
    hd = input_data['hidden_dim']
    bsz = input_data['batch_size']
    seq = input_data['seq_length']
    ci = input_data['checkpoint_interval']
    attn_heads = input_data['attn_heads']
    peak_tflops = input_data.get('peak_tflops', 70)

    # Total computation per iteration
    computation_per_iter = 2 * 4 * 12 * bsz * seq * nl * hd ** 2

    # AIT w.r.t. Parameters and Gradients
    ait_params_gradients = seq * bsz

    # AIT w.r.t. Optimizer States
    ait_optimizer_states = seq * bsz / 4

    # AIT w.r.t. Activation Checkpoints
    ait_activation_checkpoints = 24 * hd * ci

    # Efficiency calculation
    def calculate_efficiency(ait, bw):
        return (ait * bw) / (ait * bw + peak_tflops)

    # Bandwidth range for analysis (in specified unit)
    unit = input_data.get('unit', 'GB')
    if unit == 'TB':
        bandwidth_range = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        bandwidth_unit = 'TB/s'
    elif unit == 'GB':
        bandwidth_range = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        bandwidth_unit = 'GB/s'
    elif unit == 'MB':
        bandwidth_range = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        bandwidth_unit = 'MB/s'
    elif unit == 'KB':
        bandwidth_range = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
        bandwidth_unit = 'KB/s'
    else:
        print(f"Error: Unit {unit} not available.")
        exit(-1)

    # Calculate efficiency for each bandwidth value
    efficiency_params_gradients = [calculate_efficiency(ait_params_gradients, bw) for bw in bandwidth_range]
    efficiency_optimizer_states = [calculate_efficiency(ait_optimizer_states, bw) for bw in bandwidth_range]
    efficiency_activation_checkpoints = [calculate_efficiency(ait_activation_checkpoints, bw) for bw in bandwidth_range]

    return {
        'bandwidth_range': bandwidth_range,
        'bandwidth_unit': bandwidth_unit,
        'efficiency_params_gradients': efficiency_params_gradients,
        'efficiency_optimizer_states': efficiency_optimizer_states,
        'efficiency_activation_checkpoints': efficiency_activation_checkpoints
    }

def update_bandwidth_requirements():
    selected_file = file_var.get()
    input_file = os.path.join("models", selected_file)

    with open(input_file, 'r') as file:
        input_data = json.load(file)

    bandwidth_requirements = calculate_bandwidth_requirements(input_data)

    # Update the bandwidth requirement labels
    for i, bw in enumerate(bandwidth_requirements['bandwidth_range']):
        bandwidth_value_labels[i].configure(text=f"{bw} {bandwidth_requirements['bandwidth_unit']}")
        params_gradients_labels[i].configure(text=f"{bandwidth_requirements['efficiency_params_gradients'][i]:.2f}")
        optimizer_states_labels[i].configure(text=f"{bandwidth_requirements['efficiency_optimizer_states'][i]:.2f}")
        activation_checkpoints_labels[i].configure(text=f"{bandwidth_requirements['efficiency_activation_checkpoints'][i]:.2f}")

# Create the main window
ctk.set_appearance_mode("dark")  # Set the appearance mode to dark
ctk.set_default_color_theme("blue")  # Set the default color theme to blue

window = ctk.CTk()
window.title("Bandwidth Requirements Calculator")
window.geometry("800x600")  # Set the window size

# Create a frame for the input file selection
input_frame = ctk.CTkFrame(window)
input_frame.pack(pady=20)

file_var = ctk.StringVar()
file_label = ctk.CTkLabel(input_frame, text="Select input file:")
file_label.pack(side=ctk.LEFT, padx=(0, 10))

file_dropdown = ctk.CTkComboBox(input_frame, variable=file_var, values=os.listdir("models"))
file_dropdown.pack(side=ctk.LEFT, padx=(0, 10))

calculate_button = ctk.CTkButton(input_frame, text="Calculate", command=update_bandwidth_requirements)
calculate_button.pack(side=ctk.LEFT)

# Create a frame for the bandwidth requirements
bandwidth_frame = ctk.CTkFrame(window)
bandwidth_frame.pack(pady=20)

# Labels for bandwidth values
bandwidth_label = ctk.CTkLabel(bandwidth_frame, text="Bandwidth")
bandwidth_label.grid(row=0, column=0, padx=5, pady=5)
params_gradients_label = ctk.CTkLabel(bandwidth_frame, text="Params & Gradients")
params_gradients_label.grid(row=0, column=1, padx=5, pady=5)
optimizer_states_label = ctk.CTkLabel(bandwidth_frame, text="Optimizer States")
optimizer_states_label.grid(row=0, column=2, padx=5, pady=5)
activation_checkpoints_label = ctk.CTkLabel(bandwidth_frame, text="Activation Checkpoints")
activation_checkpoints_label.grid(row=0, column=3, padx=5, pady=5)

# Labels for efficiency values
bandwidth_value_labels = []
params_gradients_labels = []
optimizer_states_labels = []
activation_checkpoints_labels = []

for i in range(17):
    bandwidth_value_label = ctk.CTkLabel(bandwidth_frame, text="")
    bandwidth_value_label.grid(row=i+1, column=0, padx=5, pady=5)
    bandwidth_value_labels.append(bandwidth_value_label)
    
    params_gradients_efficiency_label = ctk.CTkLabel(bandwidth_frame, text="0.00")
    params_gradients_efficiency_label.grid(row=i+1, column=1, padx=5, pady=5)
    params_gradients_labels.append(params_gradients_efficiency_label)
    
    optimizer_states_efficiency_label = ctk.CTkLabel(bandwidth_frame, text="0.00")
    optimizer_states_efficiency_label.grid(row=i+1, column=2, padx=5, pady=5)
    optimizer_states_labels.append(optimizer_states_efficiency_label)
    
    activation_checkpoints_efficiency_label = ctk.CTkLabel(bandwidth_frame, text="0.00")
    activation_checkpoints_efficiency_label.grid(row=i+1, column=3, padx=5, pady=5)
    activation_checkpoints_labels.append(activation_checkpoints_efficiency_label)

# Start the main event loop
window.mainloop()

## Calculator 2
# Create the main window
ctk.set_appearance_mode("dark")  # Set the appearance mode to dark
ctk.set_default_color_theme("blue")  # Set the default color theme to blue

window = ctk.CTk()
window.title("Memory Requirements Calculator")
window.geometry("800x600")  # Set the window size

# Create a frame for the input file selection
input_frame = ctk.CTkFrame(window)
input_frame.pack(pady=20)

file_var = ctk.StringVar()
file_label = ctk.CTkLabel(input_frame, text="Select input file:")
file_label.pack(side=ctk.LEFT, padx=(0, 10))

file_dropdown = ctk.CTkComboBox(input_frame, variable=file_var, values=os.listdir("models"))
file_dropdown.pack(side=ctk.LEFT, padx=(0, 10))

load_button = ctk.CTkButton(input_frame, text="Load", command=load_input_file)
load_button.pack(side=ctk.LEFT, padx=(0, 10))

calculate_button = ctk.CTkButton(input_frame, text="Calculate", command=update_memory_requirements)
calculate_button.pack(side=ctk.LEFT)

# Create a frame for the input fields
input_fields_frame = ctk.CTkFrame(window)
input_fields_frame.pack(pady=10)

num_layers_label = ctk.CTkLabel(input_fields_frame, text="Number of Layers:")
num_layers_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.E)
num_layers_entry = ctk.CTkEntry(input_fields_frame)
num_layers_entry.grid(row=0, column=1, padx=5, pady=5)

hidden_dim_label = ctk.CTkLabel(input_fields_frame, text="Hidden Dimension:")
hidden_dim_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.E)
hidden_dim_entry = ctk.CTkEntry(input_fields_frame)
hidden_dim_entry.grid(row=1, column=1, padx=5, pady=5)

batch_size_label = ctk.CTkLabel(input_fields_frame, text="Batch Size:")
batch_size_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.E)
batch_size_entry = ctk.CTkEntry(input_fields_frame)
batch_size_entry.grid(row=2, column=1, padx=5, pady=5)

seq_length_label = ctk.CTkLabel(input_fields_frame, text="Sequence Length:")
seq_length_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.E)
seq_length_entry = ctk.CTkEntry(input_fields_frame)
seq_length_entry.grid(row=3, column=1, padx=5, pady=5)

checkpoint_interval_label = ctk.CTkLabel(input_fields_frame, text="Checkpoint Interval:")
checkpoint_interval_label.grid(row=4, column=0, padx=5, pady=5, sticky=tk.E)
checkpoint_interval_entry = ctk.CTkEntry(input_fields_frame)
checkpoint_interval_entry.grid(row=4, column=1, padx=5, pady=5)

attn_heads_label = ctk.CTkLabel(input_fields_frame, text="Attention Heads:")
attn_heads_label.grid(row=5, column=0, padx=5, pady=5, sticky=tk.E)
attn_heads_entry = ctk.CTkEntry(input_fields_frame)
attn_heads_entry.grid(row=5, column=1, padx=5, pady=5)

unit_label = ctk.CTkLabel(input_fields_frame, text="Unit:")
unit_label.grid(row=6, column=0, padx=5, pady=5, sticky=tk.E)
unit_var = ctk.StringVar(value="GB")
unit_dropdown = ctk.CTkComboBox(input_fields_frame, variable=unit_var, values=["TB", "GB", "MB", "KB"])
unit_dropdown.grid(row=6, column=1, padx=5, pady=5)

gpu_memory_label = ctk.CTkLabel(input_fields_frame, text="GPU Memory (GB):")
gpu_memory_label.grid(row=7, column=0, padx=5, pady=5, sticky=tk.E)
gpu_memory_entry = ctk.CTkEntry(input_fields_frame)
gpu_memory_entry.grid(row=7, column=1, padx=5, pady=5)

# Create a frame for the memory requirements output
output_frame = ctk.CTkFrame(window)
output_frame.pack(pady=20)

model_states_var = tk.StringVar()
activation_var = tk.StringVar()
activation_checkpoints_var = tk.StringVar()
mswm_var = tk.StringVar()
awm_var = tk.StringVar()
num_gpus_var = tk.StringVar()

model_states_label = ctk.CTkLabel(output_frame, text="Model States:")
model_states_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.E)
model_states_value = ctk.CTkLabel(output_frame, textvariable=model_states_var)
model_states_value.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

activation_label = ctk.CTkLabel(output_frame, text="Activation Memory:")
activation_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.E)
activation_value = ctk.CTkLabel(output_frame, textvariable=activation_var)
activation_value.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

activation_checkpoints_label = ctk.CTkLabel(output_frame, text="Activation Checkpoints:")
activation_checkpoints_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.E)
activation_checkpoints_value = ctk.CTkLabel(output_frame, textvariable=activation_checkpoints_var)
activation_checkpoints_value.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

mswm_label = ctk.CTkLabel(output_frame, text="Model State Working Memory (MSWM):")
mswm_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.E)
mswm_value = ctk.CTkLabel(output_frame, textvariable=mswm_var)
mswm_value.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

awm_label = ctk.CTkLabel(output_frame, text="Activation Working Memory (AWM):")
awm_label.grid(row=4, column=0, padx=5, pady=5, sticky=tk.E)
awm_value = ctk.CTkLabel(output_frame, textvariable=awm_var)
awm_value.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)

num_gpus_label = ctk.CTkLabel(output_frame, text="Number of GPUs Required:")
num_gpus_label.grid(row=5, column=0, padx=5, pady=5, sticky=tk.E)
num_gpus_value = ctk.CTkLabel(output_frame, textvariable=num_gpus_var)
num_gpus_value.grid(row=5, column=1, padx=5, pady=5, sticky=tk.W)

# Bind the Enter key to the update function
window.bind("<Return>", update_memory_requirements)

# Start the main event loop
window.mainloop()
