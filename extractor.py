import os
import json
from datetime import datetime
import urllib3
from typing import List
from pydantic import BaseModel, Field
from llama_index.core import SimpleDirectoryReader
from llama_index.core.program import LLMTextCompletionProgram 
from config import llm, PROJECT_PATH

class Decision(BaseModel):
    title: str = Field(description="success title")
    summary: str = Field(description="short summary")
    date: str = Field(description="date")

class Rule(BaseModel):
    rule: str = Field(description="code rule")
    scope: str = Field(description="scope of the rule")

class WarningItem(BaseModel):
    message: str = Field(description="warning message")
    severity: str = Field(description="level: high, medium, low")

class ExtractionSchema(BaseModel):
    decisions: List[Decision]
    rules: List[Rule]
    warnings: List[WarningItem]

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def run_extraction():
    print(f"🔍 start: {PROJECT_PATH}")
    
    reader = SimpleDirectoryReader(input_dir=PROJECT_PATH, recursive=True, required_exts=[".md"])
    documents = reader.load_data()
    
    if not documents:
        print("❌ files not found.")
        return

    all_extracted_items = {
        "generated_at": datetime.now().isoformat(),
        "items": {"decisions": [], "rules": [], "warnings": []}
    }

    program = LLMTextCompletionProgram.from_defaults(
        output_cls=ExtractionSchema,
        prompt_template_str="Extract technical entities (decisions, rules, warnings) from the following text:\n{text}",
        llm=llm
    )

    for doc in documents:
        file_name = os.path.basename(doc.metadata.get('file_path', 'unknown'))
        print(f"📄 processing: {file_name}...")
        
        try:
            structured_response = program(text=doc.text)
            data = structured_response.dict()

            for category in ["decisions", "rules", "warnings"]:
                for item in data.get(category, []):
                    item["source_file"] = file_name
                    file_stats = os.stat(doc.metadata['file_path'])
                    item["observed_at"] = datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                    all_extracted_items["items"][category].append(item)
                    
        except Exception as e:
            print(f"⚠️ failed to process {file_name}: {e}")

    # save to JSON
    with open("structured_data.json", "w", encoding="utf-8") as f:
        json.dump(all_extracted_items, f, indent=4, ensure_ascii=False)
    
    print(f"✅ the extracted data is saved to structured_data.json")

if __name__ == "__main__":
    run_extraction()