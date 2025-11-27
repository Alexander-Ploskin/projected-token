import os
import re
import json
import logging
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PopQADatasetPreparer:
    """
    A class to prepare the akariasai/PopQA dataset by fetching Wikipedia content
    associated with Wikidata URIs in the 's_uri' and 'o_uri' columns.
    """
    
    WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
    WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
    
    def __init__(self, output_dir: str, max_workers: int = 8, batch_size: int = 50):
        """
        Initialize the preparer with robust session handling.
        """
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Create a session with retry logic and proper headers
        self.session = requests.Session()
        
        # IMPORTANT: Wikimedia policy requires a unique User-Agent.
        # Replace 'PopQABot/1.0' with your actual bot name/email if preparing for production.
        self.session.headers.update({
            "User-Agent": "PopQADatasetPreparer/1.0 (research-project; contact@example.com)"
        })
        
        # Configure robust retries for 429 (Rate Limit) and 5xx (Server Errors)
        retries = Retry(
            total=5,
            backoff_factor=1,  # Wait 1s, 2s, 4s, 8s, 16s between retries
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def load_dataset(self) -> pd.DataFrame:
        """Loads the PopQA dataset from Hugging Face."""
        logger.info("Loading akariasai/PopQA dataset from Hugging Face...")
        dataset = load_dataset("akariasai/PopQA", split="test")
        df = dataset.to_pandas()
        logger.info(f"Loaded {len(df)} rows.")
        return df

    @staticmethod
    def extract_qid_from_uri(uri: str) -> Optional[str]:
        if not isinstance(uri, str):
            return None
        match = re.search(r'entity/(Q\d+)', uri)
        return match.group(1) if match else None

    def resolve_titles_from_wikidata(self, qids: List[str]) -> Dict[str, str]:
        """
        Resolves a list of Wikidata QIDs to English Wikipedia titles using the Wikidata API.
        Includes error handling for non-JSON responses.
        """
        if not qids:
            return {}
            
        qids = list(set([q for q in qids if q]))
        mapping = {}
        logger.info(f"Resolving {len(qids)} unique QIDs against Wikidata API...")

        for i in range(0, len(qids), self.batch_size):
            batch = qids[i:i + self.batch_size]
            ids_str = "|".join(batch)
            
            params = {
                "action": "wbgetentities",
                "ids": ids_str,
                "props": "sitelinks",
                "sitefilter": "enwiki",
                "format": "json"
            }
            
            try:
                response = self.session.get(self.WIKIDATA_API_URL, params=params, timeout=10)
                
                # Check for HTTP errors first (will trigger retry logic if 429/5xx)
                response.raise_for_status()
                
                # Safely attempt to parse JSON
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON for batch starting at {i}. Response text: {response.text[:200]}")
                    continue
                
                if "entities" in data:
                    for qid, entity in data["entities"].items():
                        if "missing" in entity:
                            continue
                        sitelinks = entity.get("sitelinks", {})
                        if "enwiki" in sitelinks:
                            mapping[qid] = sitelinks["enwiki"]["title"]
                            
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error resolving batch starting at index {i}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error resolving batch starting at index {i}: {e}")
                
        return mapping

    def download_wikipedia_page(self, title: str) -> Optional[str]:
        """
        Downloads the text content of a Wikipedia page given its title.
        """
        if not title or not isinstance(title, str):
            return None
            
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": 1,
            "exintro": 0,
            "redirects": 1
        }
        
        try:
            response = self.session.get(self.WIKIPEDIA_API_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            
            for pid, page_data in pages.items():
                if int(pid) < 0: 
                    return None
                if "extract" in page_data:
                    return page_data["extract"]
        except Exception as e:
            # Log gently; missing pages are common
            logger.debug(f"Failed to fetch content for title '{title}': {e}")
            
        return None

    def process_row(self, row_data: Tuple[int, Optional[str], Optional[str]]) -> Dict:
        idx, s_title, o_title = row_data
        results = {"index": idx, "s_content": None, "o_content": None}
        
        if s_title:
            results["s_content"] = self.download_wikipedia_page(s_title)
        if o_title:
            results["o_content"] = self.download_wikipedia_page(o_title)
            
        return results

    def run(self):
        df = self.load_dataset()
        
        logger.info("Extracting Wikidata QIDs from URIs...")
        df['s_qid'] = df['s_uri'].apply(self.extract_qid_from_uri)
        df['o_qid'] = df['o_uri'].apply(self.extract_qid_from_uri)
        
        # Gather unique QIDs for resolution
        all_s_qids = df['s_qid'].dropna().unique().tolist()
        all_o_qids = df['o_qid'].dropna().unique().tolist()
        all_qids = list(set(all_s_qids + all_o_qids))
        
        qid_to_title = self.resolve_titles_from_wikidata(all_qids)
        
        df['s_resolved_title'] = df['s_qid'].map(qid_to_title)
        df['o_resolved_title'] = df['o_qid'].map(qid_to_title)
        
        # Prioritize resolving via QID, fallback to dataset title if available
        if 's_wiki_title' in df.columns:
            df['s_final_title'] = df['s_resolved_title'].fillna(df['s_wiki_title'])
        else:
            df['s_final_title'] = df['s_resolved_title']
            
        if 'o_wiki_title' in df.columns:
            df['o_final_title'] = df['o_resolved_title'].fillna(df['o_wiki_title'])
        else:
            df['o_final_title'] = df['o_resolved_title']

        tasks = []
        for idx, row in df.iterrows():
            tasks.append((idx, row.get('s_final_title'), row.get('o_final_title')))
            
        logger.info(f"Downloading Wikipedia content for {len(tasks)} rows with {self.max_workers} workers...")
        
        s_contents = {}
        o_contents = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {executor.submit(self.process_row, task): task[0] for task in tasks}
            
            for future in tqdm(as_completed(future_to_idx), total=len(tasks), desc="Fetching pages"):
                try:
                    result = future.result()
                    idx = result['index']
                    s_contents[idx] = result['s_content']
                    o_contents[idx] = result['o_content']
                except Exception as e:
                    logger.error(f"Row processing failed for index {future_to_idx[future]}: {e}")

        df['s_wiki_content'] = df.index.map(s_contents)
        df['o_wiki_content'] = df.index.map(o_contents)
        
        output_file = os.path.join(self.output_dir, "popqa_enriched.parquet")
        logger.info(f"Saving enriched dataset to {output_file}...")
        df.to_parquet(output_file, index=False)
        logger.info("Done.")
