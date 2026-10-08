import os
import random
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_type = data.get('model_type')
        dataset = data.get('dataset')
        
        # Diabetes
        if dataset == 'diabetes':
            pred = random.choice([0, 1])
            confidence = round(random.uniform(0.6, 0.99), 2)
            result = "Dương tính (Mắc bệnh)" if pred == 1 else "Âm tính (Không mắc bệnh)"
            return jsonify({
                'success': True,
                'prediction': result,
                'confidence': confidence,
                'model_used': model_type,
                'type': 'text'
            })
            
        # Housing
        elif dataset == 'housing':
            pred = round(random.uniform(2.0, 15.0), 2)
            confidence = round(random.uniform(0.7, 0.95), 2)
            result = f"{pred} Tỷ VNĐ"
            return jsonify({
                'success': True,
                'prediction': result,
                'confidence': confidence,
                'model_used': model_type,
                'type': 'text'
            })
            
        # Amazon & Gold (Time-series)
        elif dataset in ['amazon', 'gold']:
            base_price = 150.0 if dataset == 'amazon' else 82.5
            volatility = 5.0 if dataset == 'amazon' else 0.8
            
            today = datetime.now()
            
            # Generate 31 days of past data (-30 to 0)
            past_data = []
            labels = []
            current = base_price
            for i in range(31):
                current += random.uniform(-volatility, volatility)
                past_data.append(round(current, 2))
                target_date = today + timedelta(days=-30 + i)
                labels.append(target_date.strftime("%d/%m"))
                
            # Generate 7 days of future data (1 to 7)
            future_data = [None] * 30 + [past_data[-1]] # Connect the line
            for i in range(1, 8):
                current += random.uniform(-volatility, volatility)
                future_data.append(round(current, 2))
                target_date = today + timedelta(days=i)
                labels.append(target_date.strftime("%d/%m"))
                
            confidence = round(random.uniform(0.7, 0.95), 2)
            if dataset == 'gold':
                result = f"Dự báo Vàng SJC (Triệu VNĐ/Lượng)"
            else:
                result = f"Dự báo Cổ phiếu Amazon (USD)"
            
            return jsonify({
                'success': True,
                'prediction': result,
                'confidence': confidence,
                'model_used': model_type,
                'type': 'chart',
                'chartData': {
                    'labels': labels,
                    'past': past_data + [None] * 7,
                    'future': future_data
                }
            })
            
        else:
            return jsonify({'success': False, 'error': 'Loại dự đoán không hợp lệ'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
