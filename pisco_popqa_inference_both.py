import json
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import wikipedia
from tqdm import tqdm
import logging
from typing import List, Dict, Any
import time
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PISCOProcessor:
    def __init__(self, model_name: str = "naver/pisco-mistral", device: str = "cuda"):
        logger.info(f"Loading model from {model_name}...")
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Load tokenizer (using the backbone model's tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Model loaded on {self.device}")
    
    def fetch_wikipedia_text(self, title: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                page = wikipedia.page(title, auto_suggest=False)
                return page.content
            except wikipedia.exceptions.DisambiguationError as e:
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    return page.content
                except:
                    return f"Wikipedia page for {title} (disambiguated as {e.options[0]})"
            except wikipedia.exceptions.PageError:
                logger.warning(f"Page not found for title: {title}")
                return ""
            except Exception as e:
                logger.warning(f"Error fetching Wikipedia for {title}: {str(e)}")
                time.sleep(1)  # Wait before retry
        
        return ""
    
    def truncate_to_tokens(self, text: str, max_tokens: int = 128) -> str:
        tokens = self.tokenizer.encode(text, truncation=True, max_length=max_tokens)
        truncated_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return truncated_text
    
    def run_qa_inference(self, question: str, context: str) -> str:
        try:
            documents = [[context]]
            questions = [question]
            
            output = self.model.generate_from_text(
                questions=questions,
                documents=documents,
                max_new_tokens=64
            )
            
            return output[0] if isinstance(output, list) else str(output)
        except Exception as e:
            logger.error(f"Error during QA inference: {str(e)}")
            return ""
    
    def compress_and_reconstruct(self, text: str) -> Dict[str, str]:
        try:
            truncated_text = self.truncate_to_tokens(text, max_tokens=128)
            
            # Step 1: Compress the document
            embeddings = self.model.compress_documents(documents=[truncated_text])
            
            # Step 2: Create a prompt for reconstruction
            reconstruction_prompt = f"Could you give me a different version of the background sentences above?"
            
            # Step 3: Generate reconstruction from compressed documents
            reconstructed_output = self.model.generate_from_compressed_documents_and_questions(
                questions=[reconstruction_prompt],
                compressed_documents=embeddings,
                max_new_tokens=400
            )
            
            return {
                "source_text": truncated_text,
                "reconstructed_text": reconstructed_output[0] if isinstance(reconstructed_output, list) else str(reconstructed_output)
            }
        except Exception as e:
            logger.error(f"Error during compression/reconstruction: {str(e)}")
            return {"source_text": text[:500], "reconstructed_text": ""}

def main():
    processor = PISCOProcessor()
   
    dataset = pd.read_parquet('/home/mikheev/COCOM/popqa_enriched.parquet', engine='pyarrow')
    # Task 1: Question Answering on full PopQA dataset


    logger.info("Starting Task 1: Question Answering inference...")
    qa_results = []
   
    for i, example in tqdm(dataset.iterrows(), total=len(dataset), desc="Processing QA"):
        try:
            # Fetch Wikipedia text for context
            #wiki_title = example.get('s_wiki_title', '')
            wiki_title = example['s_wiki_title']
            #if not wiki_title:
            #    wiki_title = example.get('subj', '')  # Fallback to subject name
            
            #wiki_text = processor.fetch_wikipedia_text(wiki_title)
            wiki_text = example['s_wiki_content']
            # Get question and true answer
            #question = example.get('question', '')
            #true_answer = example.get('obj', '')

            question = example['question']
            true_answer = example['obj']
            
            if wiki_text and question:
                model_answer = processor.run_qa_inference(question, wiki_text)
                
                qa_results.append({
                    "id": example['id'],
                    "question": question,
                    "true_answer": true_answer,
                    "possible_answers": example['possible_answers'],
                    "wiki_title": wiki_title,
                    "model_answer": model_answer,
                    "context_used": wiki_text
                })
            else:
                logger.warning(f"Skipping example {i} due to missing data")
                
        except Exception as e:
            logger.error(f"Error processing example {i}: {str(e)}")
            return
            continue
        
        if (i + 1) % 100 == 0:
            with open(f"pisco_popqa_qa_results_batch_{i//100}.jsonl", "w") as f:
                for result in qa_results[-100:]:
                    f.write(json.dumps(result) + "\n")
    
    
    with open("pisco_popqa_qa_results.jsonl", "w") as f:
        for result in qa_results:
            f.write(json.dumps(result) + "\n")
    
    logger.info(f"Saved {len(qa_results)} QA results to pisco_popqa_qa_results.jsonl")
    
    # Task 2: Compression and Reconstruction
    logger.info("Starting Task 2: Compression and Reconstruction...")
    compression_results = []
    
    for i, example in tqdm(dataset.iterrows(), total=len(dataset), desc="Processing Compression"):
        try:
            wiki_title = example['s_wiki_title']
            if not wiki_title:
                continue
            
            wiki_text = example['s_wiki_content']
            if wiki_text:
                result = processor.compress_and_reconstruct(wiki_text)
                
                compression_results.append({
                    "id": example['id'],
                    "wiki_title": wiki_title,
                    "subject": example['subj'],
                    "full_wiki_text_length": len(wiki_text),
                    **result
                })
            else:
                logger.warning(f"No Wikipedia text found for {wiki_title}")
                
        except Exception as e:
            logger.error(f"Error in compression for example {i}: {str(e)}")
            continue
        
        if (i + 1) % 50 == 0:
            with open(f"pisco_compression_results_batch_{(i//50)+1}.jsonl", "w") as f:
                for result in compression_results[-50:]:
                    f.write(json.dumps(result) + "\n")
   
    with open("pisco_compression_reconstruction_results.jsonl", "w") as f:
        for result in compression_results:
            f.write(json.dumps(result) + "\n")
    
    logger.info(f"Saved {len(compression_results)} compression results to pisco_compression_reconstruction_results.jsonl")
    

if __name__ == "__main__":
    
    main()
