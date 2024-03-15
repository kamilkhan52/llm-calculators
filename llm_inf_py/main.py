from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    default_input = {
        'precision': 2,  # fp16 inference
        'nlayers': 48,
        'dmodel': 7168,
        'seqlen': 1024,
        'batch': 128
    }

    if request.method == 'POST':
        if 'reset' in request.form:
            input_data = default_input
        else:
            input_data = {
                'precision': int(request.form['precision']),
                'nlayers': int(request.form['nlayers']),
                'dmodel': int(request.form['dmodel']),
                'seqlen': int(request.form['seqlen']),
                'batch': int(request.form['batch'])
            }

        kv_cache = input_data['precision'] * input_data['nlayers'] * input_data['dmodel'] * input_data['seqlen'] * input_data['batch']
        model_size = 2 * input_data['precision'] * input_data['nlayers'] * input_data['dmodel'] * input_data['dmodel']

        kv_cache_gb = round(kv_cache / (1024 ** 3), 2)
        model_size_gb = round(model_size / (1024 ** 3), 2)

        return render_template('index.html', kv_cache=kv_cache_gb, model_size=model_size_gb, input_data=input_data, default_input=default_input)

    return render_template('index.html', default_input=default_input)

if __name__ == '__main__':
    app.run(debug=True, port = 5001)