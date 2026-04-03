from flask import Flask, request, jsonify
from pipeline.orchestrator import run_pipeline
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "AutoResearch Multi-Agent Backend is Running 🚀"

@app.route("/run", methods=["POST"])
def run():
    try:
        data = request.get_json()

        if not data or "query" not in data:
            return jsonify({"error": "Query not provided"}), 400

        query = data.get("query")

        result = run_pipeline(query)

        return jsonify({
            "result": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )