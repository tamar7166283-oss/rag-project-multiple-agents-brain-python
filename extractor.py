import os
import json
from datetime import datetime
import urllib3
from typing import List
from pydantic import BaseModel, Field # ייבוא הכלים להגדרת המבנה
from llama_index.core import SimpleDirectoryReader
from llama_index.core.program import LLMTextCompletionProgram # הכלי שמחבר בין ה-LLM לסכמה
from config import llm, PROJECT_PATH

# 1. הגדרת המבנה (כאן זה המקום!)
class Decision(BaseModel):
    title: str = Field(description="כותרת ההחלטה")
    summary: str = Field(description="סיכום קצר של ההחלטה")
    date: str = Field(description="תאריך (אם קיים)")

class Rule(BaseModel):
    rule: str = Field(description="תוכן הכלל")
    scope: str = Field(description="היקף הכלל (UI, Backend וכו')")

class WarningItem(BaseModel):
    message: str = Field(description="תוכן האזהרה")
    severity: str = Field(description="רמת חומרה: high, medium, low")

class ExtractionSchema(BaseModel):
    decisions: List[Decision]
    rules: List[Rule]
    warnings: List[WarningItem]

# 2. נטרול אזהרות נטפרי
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def run_extraction():
    print(f"🔍 מתחיל סריקה בנתיב: {PROJECT_PATH}")
    
    reader = SimpleDirectoryReader(input_dir=PROJECT_PATH, recursive=True, required_exts=[".md"])
    documents = reader.load_data()
    
    if not documents:
        print("❌ לא נמצאו קבצים.")
        return

    all_extracted_items = {
        "generated_at": datetime.now().isoformat(),
        "items": {"decisions": [], "rules": [], "warnings": []}
    }

    # 3. הגדרת ה"תוכנית" שמשתמשת בסכמה
    program = LLMTextCompletionProgram.from_defaults(
        output_cls=ExtractionSchema,
        prompt_template_str="Extract technical entities (decisions, rules, warnings) from the following text:\n{text}",
        llm=llm
    )

    for doc in documents:
        file_name = os.path.basename(doc.metadata.get('file_path', 'unknown'))
        print(f"📄 מעבד: {file_name}...")
        
        try:
            # חילוץ מובנה ישירות לאובייקט פייתון
            structured_response = program(text=doc.text)
            data = structured_response.dict()

            for category in ["decisions", "rules", "warnings"]:
                for item in data.get(category, []):
                    item["source_file"] = file_name
                    # הוספת זמן אבחנה
                    file_stats = os.stat(doc.metadata['file_path'])
                    item["observed_at"] = datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                    all_extracted_items["items"][category].append(item)
                    
        except Exception as e:
            print(f"⚠️ שגיאה ב-{file_name}: {e}")

    # שמירה ל-JSON
    with open("structured_data.json", "w", encoding="utf-8") as f:
        json.dump(all_extracted_items, f, indent=4, ensure_ascii=False)
    
    print(f"✅ הסריקה הושלמה! הנתונים נשמרו ב-structured_data.json")

if __name__ == "__main__":
    run_extraction()