from flask import Flask, render_template, request
import os
import json

app = Flask(__name__)

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
        bandwidth_range = [bw/1024 for bw in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]]
        bandwidth_unit = 'TB/s'
    elif unit == 'GB':
        bandwidth_range = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        bandwidth_unit = 'GB/s'
    elif unit == 'MB':
        bandwidth_range = [bw*1024 for bw in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]]
        bandwidth_unit = 'MB/s'
    elif unit == 'KB':
        bandwidth_range = [bw*1024**2 for bw in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]]
        bandwidth_unit = 'KB/s'
    else:
        print(f"Error: Unit {unit} not available.")
        exit(-1)

    bandwidth_range_tb = [bw/1024 for bw in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]]
    # Calculate efficiency for each bandwidth value
    efficiency_params_gradients = [calculate_efficiency(ait_params_gradients, bw) for bw in bandwidth_range_tb]
    efficiency_optimizer_states = [calculate_efficiency(ait_optimizer_states, bw) for bw in bandwidth_range_tb]
    efficiency_activation_checkpoints = [calculate_efficiency(ait_activation_checkpoints, bw) for bw in bandwidth_range_tb]

    return {
        'bandwidth_range': bandwidth_range,
        'bandwidth_unit': bandwidth_unit,
        'efficiency_params_gradients': efficiency_params_gradients,
        'efficiency_optimizer_states': efficiency_optimizer_states,
        'efficiency_activation_checkpoints': efficiency_activation_checkpoints
    }

@app.route('/', methods=['GET', 'POST'])
def home():
    files = os.listdir("models")
    input_data = None
    preloaded_file = None
    default_input = {
        'num_layers': 128,
        'hidden_dim': 25600,
        'batch_size': 1,
        'seq_length': 2048,
        'checkpoint_interval': 32,
        'attn_heads': 160,
        'unit': 'GB',
        'peak_tflops': 70
    }

    if request.method == 'POST':
        if 'reset' in request.form:
            input_data = default_input
        elif 'load' in request.form:
            selected_file = request.form['file']
            input_file = os.path.join("models", selected_file)
            with open(input_file, 'r') as file:
                input_data = json.load(file)
            preloaded_file = selected_file
        else:
            input_data = {
                'num_layers': int(request.form['num_layers']),
                'hidden_dim': int(request.form['hidden_dim']),
                'batch_size': int(request.form['batch_size']),
                'seq_length': int(request.form['seq_length']),
                'checkpoint_interval': int(request.form['checkpoint_interval']),
                'attn_heads': int(request.form['attn_heads']),
                'unit': request.form['unit'],
                'peak_tflops': float(request.form['peak_tflops'])
            }

        bandwidth_requirements = calculate_bandwidth_requirements(input_data)

        return render_template('index.html', bandwidth_requirements=bandwidth_requirements, input_data=input_data, files=files, preloaded_file=preloaded_file, default_input=default_input)
    else:
        if files:
            input_file = os.path.join("models", files[0])
            with open(input_file, 'r') as file:
                input_data = json.load(file)
            preloaded_file = files[0]
        else:
            input_data = default_input
            bandwidth_requirements = calculate_bandwidth_requirements(input_data)
            return render_template('index.html', bandwidth_requirements=bandwidth_requirements, input_data=input_data, files=files, preloaded_file=preloaded_file, default_input=default_input)

    return render_template('index.html', input_data=input_data, files=files, preloaded_file=preloaded_file, default_input=default_input)

if __name__ == '__main__':
    app.run(debug=True, port=8000)