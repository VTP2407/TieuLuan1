let predictionChart = null;

const inputsConfig = {
    diabetes: [
        { id: 'gender', label: 'Giới tính (0=Nữ, 1=Nam)' },
        { id: 'age', label: 'Tuổi' },
        { id: 'hypertension', label: 'Tăng huyết áp (0/1)' },
        { id: 'heart_disease', label: 'Bệnh tim (0/1)' },
        { id: 'smoking_history', label: 'Tiền sử hút thuốc (0-5)' },
        { id: 'bmi', label: 'Chỉ số BMI' },
        { id: 'hba1c_level', label: 'Chỉ số HbA1c' },
        { id: 'blood_glucose_level', label: 'Lượng đường huyết' }
    ],
    housing: [
        { id: 'area', label: 'Diện tích (m²)' },
        { id: 'frontage', label: 'Mặt tiền (m)' },
        { id: 'access_road', label: 'Đường vào (m)' },
        { id: 'floors', label: 'Số tầng' },
        { id: 'bedrooms', label: 'Số phòng ngủ' },
        { id: 'bathrooms', label: 'Số phòng tắm' },
        { id: 'house_direction', label: 'Hướng nhà (mã hoá 0-7)' },
        { id: 'balcony_direction', label: 'Hướng ban công' },
        { id: 'legal_status', label: 'Tình trạng pháp lý' },
        { id: 'furniture_state', label: 'Tình trạng nội thất' }
    ]
};

function updateInputs() {
    const dataset = document.getElementById('dataset').value;
    const dynamicInputs = document.getElementById('dynamic-inputs');
    dynamicInputs.innerHTML = '';

    if (dataset === 'amazon' || dataset === 'gold') {
        dynamicInputs.innerHTML = `
            <div class="info-text">
                Mô hình sẽ tự động truy xuất dữ liệu giá lịch sử 30 ngày gần nhất để dự báo xu hướng 7 ngày tiếp theo. Không cần nhập thủ công.
            </div>
        `;
    } else if (inputsConfig[dataset]) {
        inputsConfig[dataset].forEach(input => {
            dynamicInputs.innerHTML += `
                <div class="input-item">
                    <label for="${input.id}">${input.label}</label>
                    <input type="text" id="${input.id}" placeholder="Nhập số..." required>
                </div>
            `;
        });
    }
}

document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const dataset = document.getElementById('dataset').value;
    const modelType = document.querySelector('input[name="model_type"]:checked').value;
    
    // Gather features if it's diabetes or housing
    let features = [];
    if (dataset === 'diabetes' || dataset === 'housing') {
        const inputs = inputsConfig[dataset];
        for (let i = 0; i < inputs.length; i++) {
            const val = document.getElementById(inputs[i].id).value;
            features.push(parseFloat(val) || 0);
        }
    }

    const btn = document.getElementById('submit-btn');
    const resultContainer = document.getElementById('result-container');
    const resultValue = document.getElementById('result-value');
    const resultConfidence = document.getElementById('result-confidence');
    const modelBadge = document.getElementById('model-badge');
    const chartContainer = document.getElementById('chart-container');

    // UI Loading state
    btn.classList.add('loading');
    btn.disabled = true;
    resultContainer.classList.add('hidden');
    chartContainer.classList.add('hidden');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset: dataset,
                model_type: modelType,
                features: features
            })
        });

        const data = await response.json();

        // Simulate network delay for effect
        setTimeout(() => {
            if(data.success) {
                resultValue.textContent = data.prediction;
                resultConfidence.textContent = Math.round(data.confidence * 100);
                
                let modelLabel = '';
                if(modelType === 'scratch') modelLabel = 'NUMPY';
                if(modelType === 'keras') modelLabel = 'Keras / TF';
                if(modelType === 'pytorch') modelLabel = 'PyTorch';
                modelBadge.textContent = modelLabel;

                resultContainer.classList.remove('hidden');

                if (data.type === 'chart' && data.chartData) {
                    chartContainer.classList.remove('hidden');
                    renderChart(data.chartData);
                }

            } else {
                alert('Lỗi: ' + data.error);
            }
            
            btn.classList.remove('loading');
            btn.disabled = false;
        }, 800);

    } catch (error) {
        console.error('Lỗi:', error);
        alert('Có lỗi xảy ra khi gọi dự báo.');
        btn.classList.remove('loading');
        btn.disabled = false;
    }
});

function renderChart(chartData) {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    
    if (predictionChart) {
        predictionChart.destroy();
    }

    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.labels,
            datasets: [
                {
                    label: 'Quá khứ (Thực tế)',
                    data: chartData.past,
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    segment: {
                        borderDash: ctx => undefined // Solid line
                    }
                },
                {
                    label: 'Tương lai (Dự báo)',
                    data: chartData.future,
                    borderColor: '#f43f5e',
                    borderWidth: 2,
                    borderDash: [5, 5], // Dashed line
                    tension: 0.4,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#1e293b' }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Ngày',
                        color: '#1e293b',
                        font: {
                            weight: 'bold'
                        }
                    },
                    ticks: { 
                        color: '#475569',
                        font: { size: 10 }
                    },
                    grid: { color: 'rgba(0,0,0,0.1)' }
                },
                y: {
                    ticks: { 
                        color: '#475569',
                        font: { size: 10 }
                    },
                    grid: { color: 'rgba(0,0,0,0.1)' }
                }
            }
        }
    });
}
