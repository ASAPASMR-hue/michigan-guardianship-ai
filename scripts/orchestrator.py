import pandas as pd
import yaml
import subprocess
import json
import pathlib
import time
import pdfplumber
import os
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/orchestrator.log'),
        logging.StreamHandler()
    ]
)

# Paths - using capital Data directory as it exists
CSV = pathlib.Path("Data/guardianship_qa_cleaned - rubric_determining.csv")
KB_DIR = pathlib.Path("Data/kb")
PDF_DIR = pathlib.Path("Data/PDF Rubrics")
SYS_DIR = pathlib.Path("Data/system_instructions")
RUBRIC_YAML = pathlib.Path("rubrics/rubric.yaml")

# PDF ranges mapping - note the typo "fpr" in actual filename
PDF_RANGES = {
    range(1, 31): "Rubric for Questions 1-30.pdf",
    range(31, 61): "Rubric fpr Questions 31-60.pdf",  # typo in actual filename
    range(61, 91): "Rubrics for Questions 61–90.pdf",
    range(91, 126): "Rubric for Questions 91–125 (Michigan Minor Guardianship).pdf",
}

# Knowledge base document mapping based on category
KB_MAPPING = {
    # Common categories from CSV
    "limited": "3. Limited Guardianship of a Minor – Voluntary Arrangement (Draft 2) copy.txt",
    "full": "2. Full Guardianship of a Minor – Eligibility and Process copy.docx.txt",
    "forms": "8. Filing for Guardianship – Court Procedures and Forms (Draft 3)-2.txt",
    "general": "1. Overview of Minor Guardianships in Michigan (Draft 2)-2-2 copy.docx.txt",
    "process": "8. Filing for Guardianship – Court Procedures and Forms (Draft 3)-2.txt",
    
    # Original mappings
    "overview": "1. Overview of Minor Guardianships in Michigan (Draft 2)-2-2 copy.docx.txt",
    "full_guardianship": "2. Full Guardianship of a Minor – Eligibility and Process copy.docx.txt",
    "limited_guardianship": "3. Limited Guardianship of a Minor – Voluntary Arrangement (Draft 2) copy.txt",
    "indian_children": "4. Guardianship of Indian Children – ICWA_MIFPA Requirements (Draft 2)-2 copy.docx.txt",
    "temporary": "5. Temporary Guardianships – Emergency and Interim Care (Draft 2)-2 copy.docx.txt",
    "duties": "6. Duties and Responsibilities of a Guardian of a Minor (Draft 2) copy.txt",
    "ending": "7. Ending or Changing a Guardianship (Termination & Modification) (Draft 2)-2 copy.docx.txt",
    "filing": "8. Filing for Guardianship – Court Procedures and Forms (Draft 3)-2.txt",
    "assets": "9. Managing a Minor's Assets – Conservatorships and Financial Considerations (Draft 2) copy.docx.txt",
}


def load_system_instructions() -> str:
    """Pre-load and truncate system instruction corpus"""
    try:
        instructions = []
        for file_path in SYS_DIR.iterdir():
            if file_path.suffix == '.txt':
                content = file_path.read_text(encoding='utf-8')[:3000]
                instructions.append(f"=== {file_path.name} ===\n{content}")
        
        return "\n\n".join(instructions)
    except Exception as e:
        logging.error(f"Error loading system instructions: {e}")
        return ""


def extract_pdf_text(q_num: int) -> str:
    """Extract text from the appropriate PDF rubric based on question number"""
    for r, pdf_name in PDF_RANGES.items():
        if q_num in r:
            pdf_path = PDF_DIR / pdf_name
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    pages_text = []
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            pages_text.append(text)
                    return "\n- - -\n".join(pages_text)
            except Exception as e:
                logging.error(f"Error extracting PDF text for question {q_num}: {e}")
                return ""
    return ""


def get_kb_document(category: str) -> str:
    """Get the appropriate knowledge base document based on category"""
    # Map category to KB document - you may need to adjust this mapping
    # based on actual category values in the CSV
    kb_filename = KB_MAPPING.get(category.lower(), "")
    
    if not kb_filename:
        # Default to overview if category not found
        kb_filename = KB_MAPPING["overview"]
        logging.warning(f"Category '{category}' not found in mapping, using overview document")
    
    kb_path = KB_DIR / kb_filename
    
    try:
        return kb_path.read_text(encoding='utf-8')
    except Exception as e:
        logging.error(f"Error reading KB document for category {category}: {e}")
        return ""


def call_gemini_api(user_prompt: str, system_prompt: str) -> Dict[str, Any]:
    """Call Gemini API using the API key directly"""
    try:
        # Using the Gemini API directly with the key
        import requests
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Try different model names
        model_names = [
            "gemini-1.5-flash",
            "gemini-1.5-pro", 
            "gemini-pro",
            "gemini-1.0-pro"
        ]
        
        for model_name in model_names:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": system_prompt + "\n\n" + user_prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0,
                    "topK": 1,
                    "topP": 1,
                    "maxOutputTokens": 2048,
                }
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                logging.info(f"Successfully connected to model: {model_name}")
                break
            elif response.status_code == 404:
                logging.debug(f"Model {model_name} not found, trying next...")
                continue
            else:
                response.raise_for_status()
        else:
            # If all models failed
            raise Exception(f"All model attempts failed. Last status: {response.status_code}")
        
        result = response.json()
        
        # Extract the generated text
        generated_text = result['candidates'][0]['content']['parts'][0]['text']
        
        # Parse JSON from the response
        try:
            # Find JSON in the response
            import re
            json_match = re.search(r'\{[^{}]*\}', generated_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                logging.error(f"No JSON found in response: {generated_text}")
                return {"required_concepts": [], "required_citations": []}
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON from response: {e}")
            logging.error(f"Response text: {generated_text}")
            return {"required_concepts": [], "required_citations": []}
            
    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}")
        return {"required_concepts": [], "required_citations": []}


def main():
    """Main orchestrator function"""
    logging.info("Starting orchestrator pipeline")
    
    # Load system instructions once
    sys_instructions = load_system_instructions()
    logging.info(f"Loaded system instructions: {len(sys_instructions)} characters")
    
    # Load CSV
    try:
        rows = pd.read_csv(CSV)
        logging.info(f"Loaded {len(rows)} questions from CSV")
    except Exception as e:
        logging.error(f"Error loading CSV: {e}")
        return
    
    # Load or create rubric YAML
    if RUBRIC_YAML.exists():
        with open(RUBRIC_YAML, 'r') as f:
            rubric_yaml = yaml.safe_load(f) or {}
    else:
        rubric_yaml = {}
        # Create rubrics directory if it doesn't exist
        RUBRIC_YAML.parent.mkdir(exist_ok=True)
    
    # Load prompt templates
    system_prompt_path = pathlib.Path("scripts/prompts/gemini_system.txt")
    user_prompt_template_path = pathlib.Path("scripts/prompts/gemini_user_template.txt")
    
    if not system_prompt_path.exists() or not user_prompt_template_path.exists():
        logging.error("Prompt templates not found. Please create them first.")
        return
    
    system_prompt = system_prompt_path.read_text()
    user_prompt_template = user_prompt_template_path.read_text()
    
    # Process each question
    for idx, row in rows.iterrows():
        try:
            # Extract question number
            q_id = row['id']
            
            # Extract numeric part for PDF mapping
            # Handle both GAP and TC prefixes
            if q_id.startswith('GAP'):
                q_num = int(q_id.replace('GAP', ''))
            elif q_id.startswith('TC'):
                q_num = int(q_id.replace('TC', ''))
            elif q_id.startswith('Q'):
                q_num = int(q_id.replace('Q', ''))
            else:
                # Try to extract any numeric part
                import re
                match = re.search(r'\d+', q_id)
                if match:
                    q_num = int(match.group())
                else:
                    logging.error(f"Could not extract question number from ID: {q_id}")
                    continue
            
            logging.info(f"Processing question {q_id} ({idx + 1}/{len(rows)})")
            
            # Get KB document based on category
            kb_document = get_kb_document(row.get('category', 'overview'))
            
            # Extract PDF rubric text
            pdf_text = extract_pdf_text(q_num)
            
            # Format user prompt - note the column is 'question' not 'question_text'
            user_prompt = user_prompt_template.format(
                question=row['question'],
                kb_document=kb_document[:5000],  # Truncate if too long
                rubric_excerpt=pdf_text[:5000],   # Truncate if too long
                system_excerpt=sys_instructions
            )
            
            # Call Gemini API
            result = call_gemini_api(user_prompt, system_prompt)
            
            # Update rubric YAML
            if 'question_overrides' not in rubric_yaml:
                rubric_yaml['question_overrides'] = {}
            
            if q_id not in rubric_yaml['question_overrides']:
                rubric_yaml['question_overrides'][q_id] = {}
            
            rubric_yaml['question_overrides'][q_id]['required_concepts'] = result.get('required_concepts', [])
            rubric_yaml['question_overrides'][q_id]['required_citations'] = result.get('required_citations', [])
            
            # Save progress after each question
            with open(RUBRIC_YAML, 'w') as f:
                yaml.dump(rubric_yaml, f, sort_keys=False, default_flow_style=False)
            
            logging.info(f"Successfully processed question {q_id}")
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Error processing question {row.get('id', 'unknown')}: {e}")
            continue
    
    logging.info("Orchestrator pipeline completed")


if __name__ == "__main__":
    main()