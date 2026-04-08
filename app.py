from flask import Flask, request, jsonify
from autoquant import AutoQuantizer
import threading

app = Flask(__name__)
tasks = {}

@app.route("/api/quantize", methods=["POST"])
def quantize_api():
    data = request.json
    model_name = data["model_name"]

    task_id = str(len(tasks))
    tasks[task_id] = {"status": "running"}

    def job():
        try:
            quantizer = AutoQuantizer(model_name)
            quantizer.run()
            tasks[task_id]["status"] = "done"
        except Exception as e:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = str(e)

    threading.Thread(target=job).start()
    return jsonify({"task_id": task_id})

@app.route("/api/status/<task_id>")
def status(task_id):
    return jsonify(tasks.get(task_id, {}))

if __name__ == "__main__":
    app.run(debug=True)