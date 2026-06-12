"""
app.py
------
Tanggung jawab TUNGGAL: HTTP routing.

Tidak ada logika model, preprocessing, atau cosine di sini.
Semua diserahkan ke core/chatbot.py.
"""

import os
import sys
from flask import Flask, request, jsonify
from core.chatbot import get_response, retrain, retrain_dari_file

os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# ==========================================================
# 🔹 ENDPOINT CHATBOT
# ==========================================================
@app.route('/api/chat', methods=['POST'])
def chat():
    data       = request.json or {}
    user_input = data.get('pesan', '').strip()

    if not user_input:
        return jsonify({'error': 'Pesan kosong'}), 400

    result = get_response(user_input)
    return jsonify(result)


# ==========================================================
# 🔹 ENDPOINT RETRAIN
# ==========================================================
@app.route('/api/retrain', methods=['POST'])
def retrain_endpoint():
    try:
        data_json = request.json
        if not data_json:
            return jsonify({'status': 'error', 'pesan': 'Data JSON kosong'}), 400

        retrain(data_json)

        return jsonify({
            'status': 'success',
            'pesan' : '✅ Model berhasil diretrain!'
        })

    except Exception as e:
        print(f"[app] ❌ Error retrain: {e}")
        return jsonify({'status': 'error', 'pesan': str(e)}), 500


# ==========================================================
# 🔹 ENDPOINT STATUS
# ==========================================================
@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({'status': 'online'}), 200


# ==========================================================
# 🔹 RUN
# ==========================================================
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        print("🚀 Training manual dari terminal...")
        try:
            retrain_dari_file()
        except Exception as e:
            print(f"❌ Gagal melatih: {e}")
    else:
        app.run(debug=True, host='127.0.0.1', port=5000)
        ##port = int(os.environ.get('PORT', 8080))  # Railway inject PORT otomatis
        ##app.run(debug=False, host='0.0.0.0', port=port)