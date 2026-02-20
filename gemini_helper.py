"""
Gemini Clinical Decision Support Helper
Created: Feb 19, 2026
For Kaggle HAI-DEF Competition
"""

from google import genai
import os

class ClinicalAssistant:
    """AI-powered clinical decision support for Lu-177 PSMA therapy"""
    
    def __init__(self):
        """Initialize Gemini client"""
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("⚠️  GOOGLE_API_KEY not set! Run: set GOOGLE_API_KEY=your_key")
        
        self.client = genai.Client(api_key=api_key)
        self.model = 'gemini-2.5-flash'
        print("✅ Gemini Clinical Assistant ready!")
    
    def check_eligibility(self, kidney_dose, egfr, predicted_lu_dose, kidney_history="None"):
        """
        Check Lu-177 therapy eligibility
        
        Args:
            kidney_dose: Kidney dose from Ga-68 scan (Gy)
            egfr: Glomerular filtration rate (mL/min)
            predicted_lu_dose: Predicted Lu-177 dose (Gy)
            kidney_history: Prior kidney disease (optional)
        
        Returns:
            Clinical recommendation (str)
        """
        prompt = f"""
You are a nuclear medicine physician expert in Lu-177 PSMA radioligand therapy.

PATIENT DATA:
- Kidney absorbed dose (Ga-68 scan): {kidney_dose} Gy
- Predicted Lu-177 kidney dose: {predicted_lu_dose} Gy
- eGFR: {egfr} mL/min/1.73m²
- Prior kidney disease: {kidney_history}

QUESTION: Is this patient eligible for Lu-177 PSMA therapy according to EANM guidelines?

Provide structured response:
1. Eligibility (Yes/No/With modifications)
2. Key considerations
3. Dose recommendations
4. Monitoring requirements
"""
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        
        return response.text
    
    def explain_to_patient(self, kidney_dose, is_safe=True):
        """
        Generate patient-friendly explanation
        
        Args:
            kidney_dose: Kidney dose value
            is_safe: Whether dose is within safe limits
        
        Returns:
            Plain language explanation (str)
        """
        safety = "within safe limits" if is_safe else "requires caution"
        
        prompt = f"""
You are explaining medical results to a patient in simple terms.

FINDING: The radiation dose to the kidneys from Lu-177 therapy would be {kidney_dose} Gy, which is {safety}.

Explain this to the patient in clear, non-technical language. Address:
1. What this number means
2. Why kidneys are important
3. What precautions (if any) are needed

Keep it reassuring but honest. Use simple words.
"""
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        
        return response.text


# TEST FUNCTION
def test_assistant():
    """Test the clinical assistant with example case"""
    print("\n" + "="*60)
    print("TESTING CLINICAL ASSISTANT")
    print("="*60 + "\n")
    
    # Create assistant
    assistant = ClinicalAssistant()
    
    # Test case
    print("📋 Test Case: Patient with Stage 3a CKD")
    print("-" * 60)
    
    recommendation = assistant.check_eligibility(
        kidney_dose=12.5,
        egfr=45,
        predicted_lu_dose=18.3,
        kidney_history="Stage 3a CKD"
    )
    
    print(recommendation)
    print("\n" + "="*60)
    print("✅ TEST COMPLETE!")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run test when script is executed
    test_assistant()