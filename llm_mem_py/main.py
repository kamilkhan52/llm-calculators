from flask import Flask, render_template, request
import math
import os
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = Flask(__name__)

def calculate_memory_requirements(input_data):
    nl = input_data['num_layers']
    hd = input_data['hidden_dim']
    bsz = input_data['batch_size']
    seq = input_data['seq_length']
    ci = input_data['checkpoint_interval']
    attn_heads = input_data['attn_heads']
    mem_gb_per_gpu = input_data['gpu_memory_gb']

    model_states_memory = 240 * nl * hd ** 2
    activation_memory = 2 * bsz * seq * hd * nl
    activation_checkpoints_memory = activation_memory / ci
    mswm = 16 * hd ** 2
    awm = bsz * seq * ci * (16 * hd + 2 * attn_heads * seq)

    num_gpus = math.ceil(((model_states_memory + activation_memory) / 1024 ** 3) / mem_gb_per_gpu)
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
        return "Error: Unit not available."

    return {
        'model_states_memory': model_states_memory / divisor,
        'activation_memory': activation_memory / divisor,
        'activation_checkpoints_memory': activation_checkpoints_memory / divisor,
        'mswm': mswm / divisor,
        'awm': awm / divisor,
        'unit': unit,
        'num_gpus': num_gpus
    }

def create_memory_plot(memory_data):
    fig = make_subplots(rows=1, cols=1)

    labels = ['Model States', 'Activation', 'Activation Checkpoints', 'MSWM', 'AWM']
    values = [memory_data['model_states_memory'], memory_data['activation_memory'],
              memory_data['activation_checkpoints_memory'], memory_data['mswm'], memory_data['awm']]

    fig.add_trace(go.Bar(x=values, y=labels, orientation='h', text=values, textposition='auto'), row=1, col=1)

    fig.update_layout(
        title='Memory Components',
        xaxis_title=f'Memory ({memory_data["unit"]})',
        yaxis_title='Component',
        height=600,  # Increase the height to make the plot bigger
        width=800,   # Increase the width to make the plot bigger
        legend=dict(x=0.02, y=1, orientation='h'),
        uniformtext_minsize=12,  # Minimum font size for data labels
        uniformtext_mode='hide'  # Hide data labels if they don't fit
    )

    # Customize the data label appearance
    fig.update_traces(
        texttemplate='%{text:.2f}',  # Display values with 2 decimal places
        textfont=dict(size=14)  # Increase the font size of data labels
    )

    return fig.to_html(full_html=False)

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
        'unit': 'TB',
        'gpu_memory_gb': 80
    }

    if request.method == 'POST':
        if 'reset' in request.form:
            input_data = default_input
            memory_requirements = calculate_memory_requirements(input_data)
            plot_html = create_memory_plot(memory_requirements)
            num_gpus_required = memory_requirements['num_gpus']
            return render_template('index.html', memory_requirements=memory_requirements, num_gpus_required=num_gpus_required, input_data=input_data, files=files, plot_html=plot_html, preloaded_file=preloaded_file, default_input=default_input)
        elif 'load' in request.form:
            selected_file = request.form['file']
            input_file = os.path.join("models", selected_file)
            with open(input_file, 'r') as file:
                input_data = json.load(file)
            preloaded_file = selected_file
        # elif 'calculate' in request.form:
        input_data = {
            'num_layers': int(request.form['num_layers']),
            'hidden_dim': int(request.form['hidden_dim']),
            'batch_size': int(request.form['batch_size']),
            'seq_length': int(request.form['seq_length']),
            'checkpoint_interval': int(request.form['checkpoint_interval']),
            'attn_heads': int(request.form['attn_heads']),
            'unit': request.form['unit'],
            'gpu_memory_gb': float(request.form['gpu_memory_gb'])
        }
        memory_requirements = calculate_memory_requirements(input_data)
        plot_html = create_memory_plot(memory_requirements)
        num_gpus_required = memory_requirements['num_gpus']
        return render_template('index.html', memory_requirements=memory_requirements, num_gpus_required=num_gpus_required, input_data=input_data, files=files, plot_html=plot_html, preloaded_file=preloaded_file, default_input=default_input)
    else:
        if files:
            input_file = os.path.join("models", files[0])
            with open(input_file, 'r') as file:
                input_data = json.load(file)
            preloaded_file = files[0]
        else:
            input_data = default_input
            memory_requirements = calculate_memory_requirements(input_data)
            plot_html = create_memory_plot(memory_requirements)
            num_gpus_required = memory_requirements['num_gpus']
            return render_template('index.html', memory_requirements=memory_requirements, num_gpus_required=num_gpus_required, input_data=input_data, files=files, plot_html=plot_html, preloaded_file=preloaded_file, default_input=default_input)

    return render_template('index.html', input_data=input_data, files=files, preloaded_file=preloaded_file, default_input=default_input)

if __name__ == '__main__':
    app.run(debug=True)
