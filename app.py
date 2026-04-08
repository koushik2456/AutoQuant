"""
AutoQuant - Complete Web Application
Provides UI for model quantization with real-time progress
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import json
import os
import shutil
import threading
import traceback
from datetime import datetime
import zipfile
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Configuration
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Import AutoQuant
from autoquant import AutoQuantizer

# Store task information for real-time progress
tasks = {}

# ============================================================
# Routes
# ============================================================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/models')
def list_models():
    """List all quantized models"""
    models = []
    if os.path.exists(MODELS_DIR):
        for model_dir in os.listdir(MODELS_DIR):
            model_path = os.path.join(MODELS_DIR, model_dir)
            if os.path.isdir(model_path):
                metadata_path = os.path.join(model_path, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        models.append({
                            'id': model_dir,
                            'name': metadata.get('model_name', model_dir),
                            'size_gb': metadata.get('quantized_size_gb', 0),
                            'original_size_gb': metadata.get('original_size_gb', 0),
                            'compression_ratio': metadata.get('compression_ratio', 0),
                            'date': metadata.get('date', 'Unknown')
                        })
                    except:
                        pass
    return jsonify(models)

@app.route('/api/quantize', methods=['POST'])
def quantize():
    """Start quantization process"""
    data = request.json
    model_name = data.get('model_name')
    target_size_gb = data.get('target_size_gb', 1.0)
    task_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    tasks[task_id] = {
        'status': 'starting',
        'progress': 0,
        'message': 'Initializing...',
        'logs': []
    }
    
    def run_quantization():
        try:
            _add_log(task_id, f"🚀 Starting AutoQuant for {model_name}")
            _add_log(task_id, f"📊 Target size: {target_size_gb} GB")
            
            # Step 1: Load model
            _add_log(task_id, f"📥 Loading model: {model_name}")
            tasks[task_id]['progress'] = 10
            tasks[task_id]['message'] = 'Loading model...'
            
            quantizer = AutoQuantizer(model_name)
            _add_log(task_id, f"✅ Model loaded! Original size: {quantizer.original_size:.2f} GB")
            
            # Step 2: Analyze sensitivity
            _add_log(task_id, f"🔬 Running multi-metric sensitivity analysis...")
            tasks[task_id]['progress'] = 30
            tasks[task_id]['message'] = 'Analyzing sensitivity (4 metrics)...'
            
            sensitivity = quantizer.analyze_sensitivity(num_samples=50)
            _add_log(task_id, f"✅ Analyzed {len(sensitivity)} layers")
            
            # Step 3: Create configuration
            _add_log(task_id, f"⚙️ Creating optimal bit allocation...")
            tasks[task_id]['progress'] = 50
            tasks[task_id]['message'] = 'Creating quantization plan...'
            
            config_result = quantizer.create_config(target_size_gb)
            _add_log(task_id, f"✅ Expected size: {config_result['expected_size_gb']:.2f} GB")
            _add_log(task_id, f"✅ Compression ratio: {config_result['compression_ratio']:.1f}x")
            
            # Show bit distribution
            bit_dist = config_result['config']['bit_distribution']
            _add_log(task_id, f"📊 Bit allocation plan:")
            for bits, count in sorted(bit_dist.items()):
                _add_log(task_id, f"   INT{bits}: {count} layers")
            
            # Step 4: Apply quantization
            _add_log(task_id, f"🔧 Applying quantization...")
            tasks[task_id]['progress'] = 70
            tasks[task_id]['message'] = 'Applying quantization...'
            
            config_path = f"temp_config_{task_id}.json"
            with open(config_path, 'w') as f:
                json.dump(config_result['config'], f, indent=2)
            
            temp_output = f"temp_output_{task_id}"
            quantizer.quantize(config_path, temp_output)
            
            # Step 5: Save model
            _add_log(task_id, f"💾 Saving quantized model...")
            tasks[task_id]['progress'] = 90
            tasks[task_id]['message'] = 'Saving model...'
            
            safe_name = model_name.replace('/', '_').replace('\\', '_')
            output_dir = os.path.join(MODELS_DIR, f"{safe_name}_{task_id}")
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            shutil.move(temp_output, output_dir)
            
            # Add metadata
            metadata_path = os.path.join(output_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                metadata['date'] = datetime.now().isoformat()
                metadata['target_size_gb'] = target_size_gb
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Cleanup
            if os.path.exists(config_path):
                os.remove(config_path)
            
            _add_log(task_id, f"✅ Model saved to {output_dir}")
            _add_log(task_id, f"🎉 Quantization complete! Final size: {metadata.get('quantized_size_gb', 0):.2f} GB")
            
            tasks[task_id]['status'] = 'complete'
            tasks[task_id]['progress'] = 100
            tasks[task_id]['message'] = 'Quantization complete!'
            tasks[task_id]['compression_ratio'] = config_result['compression_ratio']
            
        except Exception as e:
            error_msg = str(e)
            _add_log(task_id, f"❌ ERROR: {error_msg}")
            traceback.print_exc()
            tasks[task_id]['status'] = 'error'
            tasks[task_id]['message'] = error_msg
            tasks[task_id]['progress'] = 0
    
    def _add_log(task_id, message):
        """Add log message to task"""
        if task_id in tasks:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}"
            tasks[task_id]['logs'].append(log_entry)
            # Keep last 50 logs
            if len(tasks[task_id]['logs']) > 50:
                tasks[task_id]['logs'] = tasks[task_id]['logs'][-50:]
            tasks[task_id]['log'] = '\n'.join(tasks[task_id]['logs'])
        print(f"[{task_id}] {message}")
    
    # Run quantization in background
    thread = threading.Thread(target=run_quantization)
    thread.start()
    
    return jsonify({'task_id': task_id})

@app.route('/api/status/<task_id>')
def get_status(task_id):
    """Get quantization task status"""
    task = tasks.get(task_id, {'status': 'unknown'})
    return jsonify(task)

@app.route('/api/info/<path:model_name>')
def model_info(model_name):
    """Get model information"""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Estimate parameters
        if hasattr(config, 'num_parameters'):
            params = config.num_parameters
        elif hasattr(config, 'hidden_size'):
            hidden = config.hidden_size
            layers = config.num_hidden_layers
            vocab = config.vocab_size
            params = (4 * hidden * hidden * 4) * layers + vocab * hidden
        else:
            params = 124_000_000
        
        fp16_size = (params * 2) / (1024**3)
        
        return jsonify({
            'name': model_name,
            'parameters_m': params / 1e6,
            'fp16_size_gb': round(fp16_size, 2),
            'recommended_budget_gb': round(max(0.3, fp16_size * 0.6), 1)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/download/<model_id>')
def download_model(model_id):
    """Download quantized model as zip"""
    model_path = os.path.join(MODELS_DIR, model_id)
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found'}), 404
    
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, model_path)
                zf.write(file_path, arcname)
    
    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"{model_id}.zip"
    )

@app.route('/api/delete/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete quantized model"""
    model_path = os.path.join(MODELS_DIR, model_id)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        return jsonify({'success': True})
    return jsonify({'error': 'Model not found'}), 404

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔢 AutoQuant - Automated Mixed-Precision Quantization")
    print("="*60)
    print("\n🌐 Web UI: http://127.0.0.1:5000")
    print("📁 Quantized models saved in 'models/' directory")
    print("\n📊 Features:")
    print("   • Multi-metric sensitivity (4 methods)")
    print("   • Optimal bit allocation (knapsack)")
    print("   • Real-time progress tracking")
    print("   • Model download/delete")
    print("\n💡 Try these models:")
    print("   • gpt2 (fastest, ~0.23GB)")
    print("   • facebook/opt-1.3b (~2.6GB)")
    print("   • TinyLlama/TinyLlama-1.1B-Chat-v1.0 (~2.2GB)")
    print("\nPress Ctrl+C to stop\n")
    app.run(debug=False, host='127.0.0.1', port=5000)