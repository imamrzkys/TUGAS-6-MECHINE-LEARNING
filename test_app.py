import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'
    print("✅ Health check passed!\n")

def test_home():
    """Test home page"""
    print("Testing home page...")
    response = requests.get(BASE_URL)
    print(f"Status: {response.status_code}")
    assert response.status_code == 200
    assert "Student Placement Prediction" in response.text
    print("✅ Home page loaded!\n")

def test_prediction():
    """Test prediction endpoint"""
    print("Testing /predict endpoint...")
    
    test_data = {
        "sl_no": 1,
        "gender": "M",
        "ssc_p": 67.0,
        "ssc_b": "Others",
        "hsc_p": 91.0,
        "hsc_b": "Others",
        "hsc_s": "Commerce",
        "degree_p": 58.0,
        "degree_t": "Sci&Tech",
        "workex": "No",
        "etest_p": 55.0,
        "specialisation": "Mkt&HR",
        "mba_p": 58.8,
        "salary": 270000.0
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=test_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    assert response.status_code == 200
    assert result['success'] == True
    assert 'prediction' in result
    assert 'confidence' in result
    assert 'probabilities' in result
    
    print(f"\n🎯 Prediction: {result['prediction']}")
    print(f"📊 Confidence: {result['confidence']}%")
    print("✅ Prediction test passed!\n")

def test_model_info():
    """Test model info endpoint"""
    print("Testing /model-info endpoint...")
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status: {response.status_code}")
    result = response.json()
    
    assert response.status_code == 200
    assert 'metadata' in result
    assert 'feature_importance' in result
    
    print(f"Algorithm: {result['metadata']['algorithm']}")
    print(f"Train Accuracy: {result['metadata']['train_accuracy']:.4f}")
    print(f"Test Accuracy: {result['metadata']['test_accuracy']:.4f}")
    print(f"Features: {len(result['feature_importance'])}")
    print("✅ Model info test passed!\n")

def run_all_tests():
    """Run all tests"""
    print("="*50)
    print("RUNNING APPLICATION TESTS")
    print("="*50)
    print()
    
    try:
        test_health()
        test_home()
        test_model_info()
        test_prediction()
        
        print("="*50)
        print("✅ ALL TESTS PASSED!")
        print("="*50)
        print("\n🎉 Application is working correctly!")
        print("📱 Open http://localhost:5000 in your browser")
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to Flask app")
        print("Make sure the app is running: python app.py")
    except AssertionError as e:
        print(f"❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    run_all_tests()
