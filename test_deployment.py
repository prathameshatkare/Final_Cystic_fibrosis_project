import requests
import json

# Test backend health
print("=" * 60)
print("TESTING CFVISION DEPLOYMENT")
print("=" * 60)

# 1. Test dashboard metrics
print("\n1. Testing Dashboard Metrics...")
try:
    response = requests.get("http://localhost:8000/api/dashboard-metrics")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✓ Dashboard API working")
        print(f"   ✓ Model Accuracy: {data['summary']['final_accuracy']:.2%}")
        print(f"   ✓ F1-Score: {data['summary']['f1_score']:.4f}")
        print(f"   ✓ Total Samples: {data['summary']['total_samples']}")
    else:
        print(f"   ✗ Failed with status {response.status_code}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 2. Test prediction endpoint
print("\n2. Testing Prediction Endpoint...")
try:
    patient_data = {
        "age_months": 12,
        "family_history_cf": 1,
        "salty_skin": 1,
        "sweat_test_simulated": 78,
        "cough_type": 3,
        "ethnicity": "Caucasian",
        "newborn_screening_result": 1,
        "respiratory_infections_frequency": 3,
        "wheezing_present": 1
    }
    
    response = requests.post(
        "http://localhost:8000/api/predict",
        json=patient_data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Prediction API working")
        print(f"   ✓ CF Probability: {result['probability']:.2%}")
        print(f"   ✓ Risk Level: {result['risk_level']}")
    else:
        print(f"   ✗ Failed with status {response.status_code}")
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 3. Test frontend
print("\n3. Testing Frontend...")
try:
    response = requests.get("http://localhost:5173")
    if response.status_code == 200:
        print(f"   ✓ Frontend server running")
        print(f"   ✓ Dashboard accessible at http://localhost:5173")
    else:
        print(f"   ✗ Failed with status {response.status_code}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 4. Test edge ONNX export
print("\n4. Testing Edge Deployment Files...")
import os
if os.path.exists("models/cf_tabular_edge.onnx"):
    print(f"   ✓ ONNX model exported ({os.path.getsize('models/cf_tabular_edge.onnx') / 1024:.1f} KB)")
else:
    print(f"   ⚠ ONNX model not found (run: python export_to_onnx.py)")

if os.path.exists("models/edge_config.json"):
    print(f"   ✓ Edge config exported ({os.path.getsize('models/edge_config.json') / 1024:.1f} KB)")
else:
    print(f"   ⚠ Edge config not found")

print("\n" + "=" * 60)
print("DEPLOYMENT TEST COMPLETE")
print("=" * 60)

print("\n📌 Access Points:")
print("   • Backend API: http://localhost:8000")
print("   • API Docs: http://localhost:8000/docs")
print("   • Frontend Dashboard: http://localhost:5173")
print("\n📌 Next Steps:")
print("   • Open browser and visit http://localhost:5173")
print("   • Test the dashboard and diagnosis features")
print("   • For edge deployment: python export_to_onnx.py")
print("   • For production: See DEPLOYMENT_GUIDE.md")
