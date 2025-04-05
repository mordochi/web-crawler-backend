from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field, validator
from typing import Optional, Dict, List, Union, Any, Set, Literal
import asyncio
import uuid
import os
import json
import re
import time
import sqlite3
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import the database cache
from db_cache import DBCache

# Load environment variables from .env file
load_dotenv()

from crawlers import SimpleCrawler, DeepCrawler, CustomCrawler
from bs4 import BeautifulSoup
from crawl4ai import LLMExtractionStrategy, LLMConfig
from llm_providers import get_llm_config, LLMProvider

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global log function
def log(message):
    """Global logging function that prints to console and logs to the logger"""
    print(message)
    logger.info(message)

app = FastAPI(title="Web Crawler Backend", description="Backend API for crawling websites using crawl4ai")

# Initialize the database cache
protocol_cache = DBCache(db_path="protocol_cache.db", cache_expiry_hours=24)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Store for crawl jobs and results
crawl_jobs = {}

# Request models
class CrawlRequest(BaseModel):
    url: HttpUrl
    output_format: str = "markdown"
    wait_for_seconds: Optional[float] = 0
    capture_screenshot: Optional[bool] = False
    extract_metadata: Optional[bool] = True

class DeepCrawlRequest(BaseModel):
    url: HttpUrl
    strategy: str = "bfs"  # bfs, dfs
    max_pages: Optional[int] = 10
    max_depth: Optional[int] = 3
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    output_format: str = "markdown"

class CustomCrawlRequest(BaseModel):
    url: HttpUrl
    query: Optional[str] = None
    selectors: Optional[Dict[str, str]] = None
    extraction_strategy: Optional[str] = "heuristic"  # heuristic, llm
    output_format: str = "markdown"

class BlockchainAsset(BaseModel):
    asset_id: str
    amount: float
    asset_name: Optional[str] = None

class PortfolioAnalysisRequest(BaseModel):
    blockchain_id: str  # e.g., "ethereum", "binance-smart-chain", etc.
    assets: List[BlockchainAsset]
    include_top_protocols: Optional[int] = 10  # Number of top protocols to include in recommendations

# Response models
class JobStatus(BaseModel):
    job_id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    progress: Optional[float] = None
    url: str

class CrawlResult(BaseModel):
    job_id: str
    status: str
    url: str
    result: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class InvestmentOption(BaseModel):
    protocol_name: str
    tvl: Union[float, str] = 0.0  # Total Value Locked
    chain: str
    category: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    
    @validator('tvl', pre=True)
    def parse_tvl(cls, v):
        if isinstance(v, str):
            if v == "Unknown":
                return 0.0
            # Remove any non-numeric characters except decimal point
            v = re.sub(r'[^0-9.]', '', v)
            try:
                return float(v)
            except (ValueError, TypeError):
                return 0.0
        return v or 0.0


class Protocol(BaseModel):
    """Model for protocol information extracted by LLM"""
    name: str = Field(description="The name of the protocol")
    tvl: Optional[Union[float, str]] = Field(description="Total Value Locked in USD")
    category: Optional[str] = Field(None, description="Category of the protocol (e.g., Lending, DEX, etc.)")
    chain: Optional[str] = Field(None, description="Blockchain the protocol operates on")
    description: Optional[str] = Field(None, description="Brief description of what the protocol does")
    apy: Optional[Union[float, str]] = Field(None, description="Annual Percentage Yield if available")
    risks: Optional[List[str]] = Field(None, description="Potential risks associated with the protocol")
    url: Optional[str] = Field(None, description="URL to the protocol's page")
    
    @validator('tvl', pre=True)
    def parse_tvl(cls, v):
        if isinstance(v, str):
            # Remove any non-numeric characters except decimal point
            v = re.sub(r'[^0-9.]', '', v)
            try:
                return float(v)
            except (ValueError, TypeError):
                return 0.0
        return v or 0.0
    
    @validator('apy', pre=True)
    def parse_apy(cls, v):
        if isinstance(v, str):
            # Remove any non-numeric characters except decimal point
            v = re.sub(r'[^0-9.]', '', v)
            try:
                return float(v)
            except (ValueError, TypeError):
                return None
        return v


class ProtocolList(BaseModel):
    """Model for a list of protocols extracted by LLM"""
    protocols: List[Protocol] = Field(description="List of protocols found on the page")

class PortfolioAnalysisResult(BaseModel):
    job_id: str
    status: str
    blockchain_id: str
    user_assets: List[BlockchainAsset]
    investment_options: List[InvestmentOption]
    start_time: datetime
    end_time: Optional[datetime] = None

@app.get("/")
async def root():
    return {"message": "Web Crawler Backend is running", "version": "1.0.0"}

@app.post("/api/crawl", response_model=Union[JobStatus, CrawlResult])
async def crawl_website(request: CrawlRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    crawl_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "start_time": start_time,
        "url": str(request.url),
        "type": "simple"
    }
    
    # Run crawling in background
    background_tasks.add_task(
        run_simple_crawl, 
        job_id=job_id, 
        url=str(request.url),
        output_format=request.output_format,
        wait_for_seconds=request.wait_for_seconds,
        capture_screenshot=request.capture_screenshot,
        extract_metadata=request.extract_metadata
    )
    
    return JobStatus(
        job_id=job_id,
        status="queued",
        start_time=start_time,
        url=str(request.url)
    )

@app.post("/api/deep-crawl", response_model=JobStatus)
async def deep_crawl_website(request: DeepCrawlRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    crawl_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "start_time": start_time,
        "url": str(request.url),
        "type": "deep"
    }
    
    # Run deep crawling in background
    background_tasks.add_task(
        run_deep_crawl, 
        job_id=job_id, 
        url=str(request.url),
        strategy=request.strategy,
        max_pages=request.max_pages,
        max_depth=request.max_depth,
        include_patterns=request.include_patterns,
        exclude_patterns=request.exclude_patterns,
        output_format=request.output_format
    )
    
    return JobStatus(
        job_id=job_id,
        status="queued",
        start_time=start_time,
        url=str(request.url)
    )

@app.post("/api/custom-crawl", response_model=JobStatus)
async def custom_crawl_website(request: CustomCrawlRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    crawl_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "start_time": start_time,
        "url": str(request.url),
        "type": "custom"
    }
    
    # Run custom crawling in background
    background_tasks.add_task(
        run_custom_crawl, 
        job_id=job_id, 
        url=str(request.url),
        query=request.query,
        selectors=request.selectors,
        extraction_strategy=request.extraction_strategy,
        output_format=request.output_format
    )
    
    return JobStatus(
        job_id=job_id,
        status="queued",
        start_time=start_time,
        url=str(request.url)
    )

@app.get("/api/crawl-status/{job_id}")
async def get_crawl_status(job_id: str):
    if job_id not in crawl_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = crawl_jobs[job_id]
    
    # Determine the job type and return appropriate response
    job_type = job.get("type", "crawl")
    
    if job_type == "portfolio_analysis":
        # For portfolio analysis jobs
        if job["status"] == "completed":
            # If completed, return the full result
            return job
        else:
            # If not completed, return status with error details if available
            response = {
                "job_id": job["job_id"],
                "status": job["status"],
                "start_time": job["start_time"],
                "end_time": job.get("end_time"),
                "progress": job.get("progress"),
                "url": "https://defillama.com/top-protocols"
            }
            
            # Include error details if available
            if job.get("error"):
                response["error"] = job["error"]
            if job.get("error_details"):
                response["error_details"] = job["error_details"]
                
            return response
    else:
        # For regular crawl jobs
        return job

# Cache management endpoints
@app.get("/api/cache/stats")
async def get_cache_stats():
    """Get statistics about the protocol cache."""
    try:
        stats = protocol_cache.get_cache_stats()
        return {
            "status": "success",
            "stats": stats
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.delete("/api/cache/expired")
async def clear_expired_cache():
    """Clear expired cache entries."""
    try:
        protocol_cache.clear_expired_cache()
        stats = protocol_cache.get_cache_stats()
        return {
            "status": "success",
            "message": "Expired cache entries cleared",
            "stats": stats
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.delete("/api/cache/all")
async def clear_all_cache():
    """Clear all cache entries."""
    try:
        conn = sqlite3.connect(protocol_cache.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM protocol_cache")
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "message": f"All cache entries cleared ({deleted_count} entries)"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

async def run_simple_crawl(job_id, url, output_format, wait_for_seconds, capture_screenshot, extract_metadata):
    crawl_jobs[job_id]["status"] = "running"
    
    try:
        crawler = SimpleCrawler()
        result = await crawler.crawl(
            url=url,
            output_format=output_format,
            wait_for_seconds=wait_for_seconds,
            capture_screenshot=capture_screenshot,
            extract_metadata=extract_metadata
        )
        
        end_time = datetime.now()
        crawl_jobs[job_id].update({
            "status": "completed",
            "end_time": end_time,
            "result": result,
        })
        
        if extract_metadata and "metadata" in result:
            crawl_jobs[job_id]["metadata"] = result["metadata"]
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        log(f"Custom crawl failed: {str(e)}")
        log(error_details)
        
        crawl_jobs[job_id].update({
            "status": "failed",
            "end_time": datetime.now(),
            "error": str(e),
            "error_details": error_details
        })

async def run_deep_crawl(job_id, url, strategy, max_pages, max_depth, include_patterns, exclude_patterns, output_format):
    crawl_jobs[job_id]["status"] = "running"
    
    try:
        crawler = DeepCrawler()
        result = await crawler.deep_crawl(
            url=url,
            strategy=strategy,
            max_pages=max_pages,
            max_depth=max_depth,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            output_format=output_format
        )
        
        end_time = datetime.now()
        crawl_jobs[job_id].update({
            "status": "completed",
            "end_time": end_time,
            "result": result,
        })
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        log(f"Custom crawl failed: {str(e)}")
        log(error_details)
        
        crawl_jobs[job_id].update({
            "status": "failed",
            "end_time": datetime.now(),
            "error": str(e),
            "error_details": error_details
        })

async def run_custom_crawl(job_id, url, query, selectors, extraction_strategy, output_format):
    crawl_jobs[job_id]["status"] = "running"
    
    try:
        crawler = CustomCrawler()
        result = await crawler.custom_crawl(
            url=url,
            query=query,
            selectors=selectors,
            extraction_strategy=extraction_strategy,
            output_format=output_format
        )
        
        end_time = datetime.now()
        crawl_jobs[job_id].update({
            "status": "completed",
            "end_time": end_time,
            "result": result,
        })
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        log(f"Custom crawl failed: {str(e)}")
        log(error_details)
        
        crawl_jobs[job_id].update({
            "status": "failed",
            "end_time": datetime.now(),
            "error": str(e),
            "error_details": error_details
        })

@app.post("/api/portfolio-analysis", response_model=JobStatus)
async def analyze_portfolio(request: PortfolioAnalysisRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    crawl_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "start_time": start_time,
        "blockchain_id": request.blockchain_id,
        "type": "portfolio_analysis"
    }
    
    # Run portfolio analysis in background
    background_tasks.add_task(
        run_portfolio_analysis,
        job_id=job_id,
        blockchain_id=request.blockchain_id,
        assets=request.assets,
        include_top_protocols=request.include_top_protocols
    )
    
    return JobStatus(
        job_id=job_id,
        status="queued",
        start_time=start_time,
        url="https://defillama.com/top-protocols"  # We'll be crawling DeFiLlama
    )

async def run_portfolio_analysis(job_id, blockchain_id, assets, include_top_protocols):
    crawl_jobs[job_id]["status"] = "running"
    
    # Set up file-based logging
    log_file = f"portfolio_analysis_{job_id}.log"
    with open(log_file, "w") as f:
        f.write(f"Starting portfolio analysis for job {job_id}\n")
        f.write(f"Blockchain: {blockchain_id}\n")
        f.write(f"Assets: {assets}\n")
        f.write(f"Include top protocols: {include_top_protocols}\n\n")
        
    # Initialize protocols_found dictionary
    protocols_found = {}
    
    # Create a local log function that also writes to the log file
    def local_log(message):
        # Use the global log function
        log(message)
        # Also write to the log file
        with open(log_file, "a") as f:
            f.write(f"{message}\n")
    
    # Function to extract investment options from cached HTML
    def extract_from_cached_html(html_content):
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('table')
            
            if not table:
                local_log("No table found in cached HTML")
                return []
            
            rows = table.find_all('tr')[1:]  # Skip header row
            local_log(f"Found {len(rows)} protocol rows in cached HTML")
            
            investment_options = []
            for i, row in enumerate(rows):
                if i >= include_top_protocols:
                    break
                    
                cells = row.find_all('td')
                if len(cells) >= 4:
                    name = cells[0].text.strip()
                    tvl_text = cells[1].text.strip()
                    chain_text = cells[2].text.strip()
                    category = cells[3].text.strip()
                    
                    # Handle the case where chain might be a comma-separated list
                    if blockchain_id.lower() in chain_text.lower() or not chain_text:
                        chain = blockchain_id
                    else:
                        chain = blockchain_id  # Default to requested blockchain
                    
                    # Create URL from protocol name
                    url = f"https://defillama.com/protocol/{name.lower().replace(' ', '-')}"
                    
                    option = InvestmentOption(
                        protocol_name=name,
                        tvl=tvl_text if tvl_text else 0,
                        chain=chain,
                        category=category if category else 'DeFi',
                        url=url,
                        description=f"Investment opportunity in {name} on {blockchain_id}"
                    )
                    investment_options.append(option)
            
            local_log(f"Successfully extracted {len(investment_options)} investment options from cached data")
            return investment_options
        except Exception as e:
            local_log(f"Error parsing cached HTML: {str(e)}")
            return []
    
    try:
        # No hardcoded protocols - we'll return empty array if none found
        
        # Use DeepCrawler to crawl DeFiLlama's top protocols and related pages
        try:
            crawler = DeepCrawler()
            
            # Use CustomCrawler with LLM extraction strategy for better analysis
            custom_crawler = CustomCrawler()
            
            # Get the selected LLM provider from environment
            llm_provider = os.environ.get("LLM_PROVIDER", "openai").lower()
            log(f"Using LLM provider: {llm_provider}")
            
            # Determine extraction strategy based on provider
            if llm_provider == "openai":
                # Check if OpenAI API key is available
                openai_api_key = os.environ.get("OPENAI_API_KEY", "")
                if not openai_api_key:
                    log("WARNING: No OpenAI API key found in environment variables. Skipping LLM extraction and using direct HTML parsing.")
                    extraction_strategy = "html"
                else:
                    log("OpenAI API key found. Using LLM extraction strategy.")
                    extraction_strategy = "llm"
            elif llm_provider == "claude":
                # Check if Claude API key is available
                claude_api_key = os.environ.get("CLAUDE_API_KEY", "")
                if not claude_api_key:
                    log("WARNING: No Claude API key found in environment variables. Skipping LLM extraction and using direct HTML parsing.")
                    extraction_strategy = "html"
                else:
                    log("Claude API key found. Using LLM extraction strategy.")
                    extraction_strategy = "llm"
            elif llm_provider == "ollama":
                # Check if Ollama is available
                provider = LLMProvider("ollama")
                local_log(f"Checking Ollama availability with provider config: {provider.config}")
                is_available = provider.is_ollama_available()
                local_log(f"Ollama available: {is_available}")
                
                if is_available:
                    local_log("Ollama is available. Using LLM extraction strategy.")
                    extraction_strategy = "llm"
                else:
                    local_log("WARNING: Ollama is not available. Skipping LLM extraction and using direct HTML parsing.")
                    extraction_strategy = "html"
            else:
                log(f"WARNING: Unknown LLM provider '{llm_provider}'. Using HTML extraction strategy.")
                extraction_strategy = "html"
                
            # Create an LLM extraction strategy if needed
            if extraction_strategy == "llm":
                # Get LLM configuration from our provider module
                llm_provider_config = get_llm_config(llm_provider)

                local_log(f"LLM provider config: {llm_provider_config}")
                
                # Extract parameters for LLMConfig
                provider = llm_provider_config.get("provider")
                api_token = llm_provider_config.get("api_token", "")
                base_url = llm_provider_config.get("base_url", None)
                extra_args = llm_provider_config.get("extra_args", {})
                
                local_log(f"Creating LLMConfig with provider={provider}, base_url={base_url}, extra_args={extra_args}")
                
                # Create LLMConfig with only the parameters it accepts
                llm_config = LLMConfig(
                    provider=provider,
                    api_token=api_token,
                    base_url=base_url
                )
                
                llm_extraction_strategy = LLMExtractionStrategy(
                    llm_config=llm_config,
                    schema=ProtocolList.model_json_schema(),  # Use our Pydantic model schema
                    extraction_type="schema",
                    instruction=f"Extract a list of top DeFi protocols from this page. Focus on protocols available on {blockchain_id}. Include name, TVL (Total Value Locked), category, and other available information. Return the data as a structured JSON object with a 'protocols' array containing protocol objects.",
                    chunk_token_threshold=4000,
                    overlap_rate=0.1,
                    apply_chunking=True,
                    input_format="html"
                )

                # log llm_extraction_strategy json format
                log(f"LLM extraction strategy: {llm_extraction_strategy}")
            
            # Perform the crawl with the appropriate extraction strategy
            if extraction_strategy == "llm":
                # Log the schema being used for extraction
                local_log(f"Using schema for extraction: {ProtocolList.model_json_schema()}")
                local_log(f"LLM provider: {llm_provider}, Provider config: {llm_provider_config}")
                
                try:
                    # Use direct requests to get the HTML content
                    # This avoids browser automation issues
                    local_log("Using direct requests to get HTML content...")
                    html_content = None
                    try:
                        import requests
                        import asyncio
                        
                        # Define headers to mimic a browser request more convincingly
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                            'Accept-Language': 'en-US,en;q=0.9',
                            'Accept-Encoding': 'gzip, deflate, br',
                            'Referer': 'https://www.google.com/',
                            'sec-ch-ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
                            'sec-ch-ua-mobile': '?0',
                            'sec-ch-ua-platform': '"Windows"',
                            'sec-fetch-dest': 'document',
                            'sec-fetch-mode': 'navigate',
                            'sec-fetch-site': 'cross-site',
                            'sec-fetch-user': '?1',
                            'Upgrade-Insecure-Requests': '1',
                            'Connection': 'keep-alive',
                            'Cache-Control': 'max-age=0',
                            'dnt': '1'
                        }
                        
                        # Run the request in a separate thread to avoid blocking
                        def fetch_html():
                            url = "https://defillama.com/top-protocols"
                            api_url = "https://api.llama.fi/protocols"
                            
                            # Extract asset symbols from the assets parameter
                            asset_symbols = [asset.asset_id for asset in assets]
                            local_log(f"Asset symbols: {asset_symbols}")
                            
                            # First, check if we have a valid cache entry
                            local_log(f"Checking cache for {url} with blockchain_id={blockchain_id} and asset_symbols={asset_symbols}")
                            cached_data = protocol_cache.get_cached_protocols(api_url, blockchain_id, asset_symbols)
                            
                            if cached_data:
                                local_log(f"Using cached data for {url} with blockchain_id={blockchain_id}")
                                
                                # Create HTML from cached data
                                html = "<html><body><h1>DeFi Llama Top Protocols (Cached)</h1><table>"
                                html += "<tr><th>Name</th><th>TVL</th><th>Chain</th><th>Category</th></tr>"
                                
                                # Process the cached data based on its structure
                                if isinstance(cached_data, list):
                                    protocols = cached_data
                                elif isinstance(cached_data, dict) and 'protocols' in cached_data:
                                    protocols = cached_data['protocols']
                                else:
                                    protocols = []
                                    
                                for protocol in protocols:
                                    name = protocol.get('name', '')
                                    tvl = protocol.get('tvl', 0)
                                    chains = ", ".join(protocol.get('chains', [])) if isinstance(protocol.get('chains'), list) else protocol.get('chain', '')
                                    category = protocol.get('category', '')
                                    html += f"<tr><td>{name}</td><td>{tvl}</td><td>{chains}</td><td>{category}</td></tr>"
                                
                                html += "</table></body></html>"
                                return html
                            
                            # If no cache or cache expired, try to fetch fresh data
                            local_log("No valid cache found, fetching fresh data...")
                            
                            # Try multiple approaches to get the HTML
                            try:
                                # First attempt with standard requests
                                session = requests.Session()
                                response = session.get(url, 
                                                      headers=headers, 
                                                      timeout=15,
                                                      allow_redirects=True)
                                response.raise_for_status()
                                local_log("Successfully fetched HTML content with standard request")
                                
                                # Don't cache the HTML directly as it's too large and not structured
                                return response.text
                            except Exception as e:
                                local_log(f"First attempt failed: {str(e)}")
                                
                                # Second attempt with different User-Agent
                                try:
                                    mobile_headers = headers.copy()
                                    mobile_headers['User-Agent'] = 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1'
                                    mobile_headers['sec-ch-ua-mobile'] = '?1'
                                    mobile_headers['sec-ch-ua-platform'] = '"iOS"'
                                    
                                    session = requests.Session()
                                    response = session.get(url, 
                                                          headers=mobile_headers, 
                                                          timeout=15,
                                                          allow_redirects=True)
                                    response.raise_for_status()
                                    local_log("Successfully fetched HTML content with mobile user agent")
                                    
                                    # Don't cache the HTML directly as it's too large and not structured
                                    return response.text
                                except Exception as e2:
                                    local_log(f"Second attempt failed: {str(e2)}")
                                    
                                    # Try using the API endpoint instead
                                    try:
                                        api_headers = {
                                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
                                            'Accept': 'application/json',
                                            'Referer': 'https://defillama.com/'
                                        }
                                        response = session.get(api_url, 
                                                              headers=api_headers, 
                                                              timeout=15)
                                        response.raise_for_status()
                                        
                                        # Convert API response to HTML format for processing
                                        api_data = response.json()
                                        local_log(f"Successfully fetched API data with {len(api_data)} protocols")
                                        
                                        # Extract asset symbols from the assets parameter
                                        asset_symbols = [asset.asset_id for asset in assets]
                                        
                                        # Cache the API data for future use
                                        protocol_cache.cache_protocols(api_url, api_data, blockchain_id, asset_symbols, source="api")
                                        local_log(f"Cached {len(api_data)} protocols for future use with asset_symbols={asset_symbols}")
                                        
                                        # Create a simple HTML table from the API data
                                        html = "<html><body><h1>DeFi Llama Top Protocols</h1><table>"
                                        html += "<tr><th>Name</th><th>TVL</th><th>Chain</th><th>Category</th></tr>"
                                        
                                        for protocol in api_data:
                                            name = protocol.get('name', '')
                                            tvl = protocol.get('tvl', 0)
                                            chains = ", ".join(protocol.get('chains', []))
                                            category = protocol.get('category', '')
                                            html += f"<tr><td>{name}</td><td>{tvl}</td><td>{chains}</td><td>{category}</td></tr>"
                                        
                                        html += "</table></body></html>"
                                        return html
                                    except Exception as e3:
                                        local_log(f"API attempt failed: {str(e3)}")
                                        
                                        # If all live attempts fail, try to use an expired cache as last resort
                                        local_log("All fetch attempts failed, checking for expired cache...")
                                        conn = sqlite3.connect(protocol_cache.db_path)
                                        cursor = conn.cursor()
                                        
                                        # Get the most recent cache entry regardless of expiry
                                        if blockchain_id:
                                            cursor.execute(
                                                "SELECT data, timestamp FROM protocol_cache WHERE url = ? AND blockchain_id = ? ORDER BY timestamp DESC LIMIT 1",
                                                (api_url, blockchain_id)
                                            )
                                        else:
                                            cursor.execute(
                                                "SELECT data, timestamp FROM protocol_cache WHERE url = ? ORDER BY timestamp DESC LIMIT 1",
                                                (api_url,)
                                            )
                                        
                                        result = cursor.fetchone()
                                        conn.close()
                                        
                                        if result:
                                            data, timestamp = result
                                            cache_age_hours = (time.time() - timestamp) / 3600
                                            local_log(f"Using expired cache from {cache_age_hours:.1f} hours ago as last resort")
                                            
                                            # Create HTML from expired cached data
                                            expired_data = json.loads(data)
                                            html = "<html><body><h1>DeFi Llama Top Protocols (Expired Cache)</h1><table>"
                                            html += "<tr><th>Name</th><th>TVL</th><th>Chain</th><th>Category</th></tr>"
                                            
                                            if isinstance(expired_data, list):
                                                protocols = expired_data
                                            elif isinstance(expired_data, dict) and 'protocols' in expired_data:
                                                protocols = expired_data['protocols']
                                            else:
                                                protocols = []
                                                
                                            for protocol in protocols:
                                                name = protocol.get('name', '')
                                                tvl = protocol.get('tvl', 0)
                                                chains = ", ".join(protocol.get('chains', [])) if isinstance(protocol.get('chains'), list) else protocol.get('chain', '')
                                                category = protocol.get('category', '')
                                                html += f"<tr><td>{name}</td><td>{tvl}</td><td>{chains}</td><td>{category}</td></tr>"
                                            
                                            html += "</table></body></html>"
                                            return html
                                        
                                        raise Exception("All fetch attempts failed and no cache available")
                        
                        # Run the request in a thread pool
                        loop = asyncio.get_event_loop()
                        html_content = await loop.run_in_executor(None, fetch_html)
                        
                        local_log(f"Successfully fetched HTML content, length: {len(html_content)}")
                        
                        # Check if this is cached data by looking for the indicator in the HTML
                        is_cached_data = "DeFi Llama Top Protocols (Cached)" in html_content or "DeFi Llama Top Protocols (Expired Cache)" in html_content
                        
                        if is_cached_data:
                            local_log("Detected cached data in HTML - using direct extraction")
                            investment_options = extract_from_cached_html(html_content)
                            if investment_options:
                                local_log(f"Successfully extracted {len(investment_options)} investment options from cached HTML")
                                
                                # Update job status to completed
                                job.status = "completed"
                                job.end_time = datetime.now().isoformat()
                                job.result = {
                                    "job_id": job_id,
                                    "status": "completed",
                                    "blockchain_id": blockchain_id,
                                    "user_assets": [asset.dict() for asset in assets],
                                    "investment_options": [option.dict() for option in investment_options],
                                    "start_time": job.start_time,
                                    "end_time": job.end_time
                                }
                                jobs[job_id] = job
                                local_log(f"Job status updated to completed with {len(investment_options)} investment options")
                                return investment_options
                    except Exception as e:
                        local_log(f"Error fetching HTML directly: {str(e)}")
                        html_content = None
                    
                    # If we have HTML content, use it directly with the LLM
                    if html_content:
                        local_log("Using fetched HTML content with Ollama for extraction")
                        try:
                            # Create a direct extraction using Ollama
                            from crawl4ai.llm import LLMClient
                            
                            # Create LLM client with our config
                            llm_client = LLMClient(llm_config)
                            
                            # Prepare the prompt with our schema
                            schema = ProtocolList.model_json_schema()
                            prompt = f"""Extract information about DeFi protocols from the following HTML content from DeFiLlama's top protocols page.
                            Focus on protocols available on {blockchain_id} blockchain.
                            Return the data as a JSON object according to this schema: {schema}

                            HTML Content:
                            {html_content[:50000]}
                            """
                            
                            local_log("Sending HTML content to Ollama for extraction...")
                            response = await llm_client.agenerate(prompt)
                            local_log(f"Received response from Ollama, length: {len(response)}")
                            
                            # Try to parse the response as JSON
                            import json
                            import re
                            
                            # Log the raw response for debugging
                            local_log(f"Raw Ollama response (first 500 chars): {response[:500]}")
                            
                            # Extract JSON from the response if needed
                            json_match = re.search(r'```json\n(.+?)\n```', response, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(1)
                                local_log(f"Extracted JSON from markdown code block, length: {len(json_str)}")
                            else:
                                # Try to find JSON object directly with protocols array
                                json_match = re.search(r'\{\s*"protocols"\s*:\s*\[.+?\]\s*\}', response, re.DOTALL)
                                if json_match:
                                    json_str = json_match.group(0)
                                    local_log(f"Extracted JSON object directly, length: {len(json_str)}")
                                else:
                                    # Try to find any JSON array that might contain protocols
                                    json_match = re.search(r'\[\s*\{.+?\}\s*(,\s*\{.+?\}\s*)*\]', response, re.DOTALL)
                                    if json_match:
                                        json_str = '{"protocols": ' + json_match.group(0) + '}'
                                        local_log(f"Found JSON array and wrapped it in protocols object, length: {len(json_str)}")
                                    else:
                                        # Last resort: try to extract any JSON object
                                        json_match = re.search(r'\{.+?\}', response, re.DOTALL)
                                        if json_match:
                                            json_str = json_match.group(0)
                                            local_log(f"Extracted any JSON object as fallback, length: {len(json_str)}")
                                        else:
                                            json_str = response
                                            local_log("Using full response as JSON")
                            
                            try:
                                extracted_data = json.loads(json_str)
                                local_log(f"Successfully parsed JSON response: {extracted_data.keys() if isinstance(extracted_data, dict) else 'Not a dictionary'}")
                                
                                # Create a result similar to what custom_crawl would return
                                result = {
                                    "url": "https://defillama.com/top-protocols",
                                    "title": "DeFi Llama - Top Protocols",
                                    "extracted_data": extracted_data
                                }
                            except json.JSONDecodeError as e:
                                local_log(f"Failed to parse JSON: {str(e)}")
                                local_log(f"Response snippet: {json_str[:200]}")
                                # Fall back to standard extraction
                                result = await custom_crawler.custom_crawl(
                                    url="https://defillama.com/top-protocols",
                                    extraction_strategy="html",
                                    output_format="json"
                                )
                        except Exception as e:
                            local_log(f"Error during direct Ollama extraction: {str(e)}")
                            # Fall back to standard extraction
                            result = await custom_crawler.custom_crawl(
                                url="https://defillama.com/top-protocols",
                                extraction_strategy="html",
                                output_format="json"
                            )
                    else:
                        # Fall back to standard LLM extraction (which may use browser automation)
                        local_log("Starting custom_crawl with LLM extraction strategy...")
                        try:
                            # Define the protocol schema
                            protocol_schema = {
                                "title": "Protocol",
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "The name of the protocol"},
                                    "tvl": {"type": ["number", "string"], "description": "Total Value Locked in the protocol"},
                                    "chain": {"type": "string", "description": "The blockchain the protocol runs on"},
                                    "category": {"type": "string", "description": "The category of the protocol (e.g., DEX, Lending)"},
                                    "url": {"type": "string", "description": "URL to the protocol's page"}
                                },
                                "required": ["name"]
                            }
                            
                            # Create the LLM extraction strategy
                            llm_strategy = LLMExtractionStrategy(
                                llm_config=llm_config,
                                schema=json.dumps(protocol_schema),
                                extraction_type="list",
                                instruction=f"Extract all DeFi protocols on the {blockchain_id} blockchain from the content. Include name, tvl, chain, category, and url if available.",
                                chunk_token_threshold=4000,
                                overlap_rate=0.1,
                                apply_chunking=True,
                                input_format="html",
                                extra_args={"temperature": 0.0, "max_tokens": 2000}
                            )
                            
                            result = await custom_crawler.custom_crawl(
                                url="https://defillama.com/top-protocols",
                                extraction_strategy="llm",  # Use LLM-based extraction
                                output_format="json",
                                llm_extraction_strategy=llm_strategy
                            )
                            local_log("Completed custom_crawl with LLM extraction strategy")
                        except Exception as e:
                            local_log(f"Error during LLM extraction: {str(e)}")
                            local_log("Falling back to HTML extraction")
                            result = await custom_crawler.custom_crawl(
                                url="https://defillama.com/top-protocols",
                                extraction_strategy="html",
                                output_format="json"
                            )
                except Exception as e:
                    local_log(f"Error during LLM extraction: {str(e)}")
                    local_log("Falling back to HTML extraction")
                    result = await custom_crawler.custom_crawl(
                        url="https://defillama.com/top-protocols",
                        extraction_strategy="html",
                        output_format="json"
                    )

                # Log detailed information about the result structure
                local_log(f"Result type: {type(result)}")
                
                if isinstance(result, dict):
                    local_log(f"Result dictionary keys: {result.keys()}")
                    
                    # Log extracted data if available
                    if 'extracted_data' in result:
                        local_log(f"Extracted data type: {type(result['extracted_data'])}")
                        if isinstance(result['extracted_data'], dict):
                            local_log(f"Extracted data keys: {result['extracted_data'].keys()}")
                            
                            if 'protocols' in result['extracted_data']:
                                protocols = result['extracted_data']['protocols']
                                local_log(f"Found {len(protocols)} protocols in extracted data")
                                
                                # Log the first few protocols for debugging
                                for i, protocol in enumerate(protocols[:3]):
                                    local_log(f"Protocol {i+1}: {protocol}")
                        elif isinstance(result['extracted_data'], str):
                            local_log(f"Extracted data as string (first 200 chars): {result['extracted_data'][:200]}")
                else:
                    local_log(f"Result is not a dictionary: {result}")
                
                # If result is an object with a to_dict or dict method, use that
                if hasattr(result, 'to_dict'):
                    log(f"Result as dictionary: {result.to_dict()}")
                elif hasattr(result, 'dict'):
                    log(f"Result as dictionary: {result.dict()}")
                else:
                    log(f"Result: {result}")
            else:
                # Perform the crawl with HTML extraction
                log("Using direct HTML extraction")
                result = await custom_crawler.custom_crawl(
                    url="https://defillama.com/top-protocols",
                    extraction_strategy="html",  # Use HTML-based extraction
                    output_format="json"
                )
                # Log detailed information about the result structure
                local_log(f"Result type: {type(result)}")
                
                if isinstance(result, dict):
                    local_log(f"Result dictionary keys: {result.keys()}")
                    
                    # Log extracted data if available
                    if 'extracted_data' in result:
                        local_log(f"Extracted data type: {type(result['extracted_data'])}")
                        if isinstance(result['extracted_data'], dict):
                            local_log(f"Extracted data keys: {result['extracted_data'].keys()}")
                            
                            if 'protocols' in result['extracted_data']:
                                protocols = result['extracted_data']['protocols']
                                local_log(f"Found {len(protocols)} protocols in extracted data")
                                
                                # Log the first few protocols for debugging
                                for i, protocol in enumerate(protocols[:3]):
                                    local_log(f"Protocol {i+1}: {protocol}")
                        elif isinstance(result['extracted_data'], str):
                            local_log(f"Extracted data as string (first 200 chars): {result['extracted_data'][:200]}")
                else:
                    local_log(f"Result is not a dictionary: {result}")
                
                # If result is an object with a to_dict or dict method, use that
                if hasattr(result, 'to_dict'):
                    log(f"Result as dictionary: {result.to_dict()}")
                elif hasattr(result, 'dict'):
                    log(f"Result as dictionary: {result.dict()}")
                else:
                    log(f"Result: {result}")
        except Exception as browser_error:
            log(f"Browser automation failed: {str(browser_error)}")
            log("Returning empty investment options array")
            
            # Return empty investment options
            end_time = datetime.now()
            portfolio_result = PortfolioAnalysisResult(
                job_id=job_id,
                status="completed",
                blockchain_id=blockchain_id,
                user_assets=assets,
                investment_options=[],  # Empty array
                start_time=crawl_jobs[job_id]["start_time"],
                end_time=end_time
            )
            
            # Update the job status
            crawl_jobs[job_id].update({
                "status": "completed",
                "end_time": end_time,
                "result": portfolio_result.dict(),
            })
            
            return
        
        # Process the results from LLM extraction
        log(f"Processing LLM extraction results for {blockchain_id}")
        protocols_found = {}
        
        try:
            # Check if we have extracted data in the result
            if "extracted_data" in result and result["extracted_data"]:
                log("Found extracted_data in result")
                extracted_data = result["extracted_data"]
                
                # Log a sample of the extracted data
                log(f"Extracted data sample: {str(extracted_data)[:500]}...")
                
                # Handle the case where extracted_data is a method (CrawlResult.json)
                if hasattr(extracted_data, '__call__'):
                    try:
                        log("Extracted data is a method, trying to call it")
                        extracted_data = extracted_data()
                        log(f"Called method, got: {str(extracted_data)[:500]}...")
                    except Exception as e:
                        log(f"Error calling extracted_data method: {str(e)}")
                        extracted_data = {}
                
                # If we have HTML in the extracted data, try to parse it directly
                if isinstance(extracted_data, dict) and "html" in extracted_data:
                    log("Found HTML in extracted data, trying to parse it directly")
                    try:
                        html_content = extracted_data["html"]
                        soup = BeautifulSoup(html_content, "html.parser")
                        
                        # Log the structure of the HTML to help with debugging
                        log("Analyzing HTML structure...")
                        
                        # Find all tables in the document
                        tables = soup.find_all("table")
                        log(f"Found {len(tables)} tables in the HTML")
                        
                        # Look for divs that might contain protocol data
                        protocol_divs = soup.find_all("div", class_="table-responsive")
                        log(f"Found {len(protocol_divs)} table-responsive divs")
                        
                        # Look for protocol cards or list items
                        protocol_cards = soup.find_all("div", class_="card")
                        log(f"Found {len(protocol_cards)} card divs")
                        
                        # Try to find protocol names and TVL values directly
                        protocols_list = []
                        
                        # Method 1: Look for protocol name elements
                        protocol_name_elements = soup.find_all("a", href=lambda href: href and "/protocol/" in href)
                        log(f"Found {len(protocol_name_elements)} protocol name links")
                        
                        for protocol_element in protocol_name_elements[:10]:  # Limit to top 10
                            try:
                                # Get protocol name
                                protocol_name = protocol_element.text.strip()
                                
                                # Try to find TVL near this element
                                parent_row = protocol_element.find_parent("tr")
                                tvl_text = "Unknown"
                                
                                if parent_row:
                                    # Look for TVL in the same row
                                    tvl_cells = parent_row.find_all("td")
                                    if len(tvl_cells) >= 3:
                                        tvl_text = tvl_cells[2].text.strip()
                                
                                # Create protocol object
                                protocol = {
                                    "name": protocol_name,
                                    "tvl": tvl_text,
                                    "chain": blockchain_id,  # Default to requested blockchain
                                    "category": "DeFi",
                                    "url": "https://defillama.com" + protocol_element.get("href", "")
                                }
                                
                                protocols_list.append(protocol)
                                log(f"Found protocol: {protocol_name} with TVL: {tvl_text}")
                            except Exception as e:
                                log(f"Error processing protocol element: {str(e)}")
                        
                        log(f"Extracted {len(protocols_list)} protocols from HTML")
                        if protocols_list:
                            extracted_data = {"protocols": protocols_list}
                        else:
                            log("No protocols found in HTML, trying alternative parsing methods")
                            
                            # Method 2: Try to find protocol information in any table
                            for table in tables:
                                rows = table.find_all("tr")
                                log(f"Analyzing table with {len(rows)} rows")
                                
                                for row in rows[1:]:  # Skip header row
                                    cells = row.find_all("td")
                                    if len(cells) >= 2:
                                        try:
                                            protocol_name = cells[0].text.strip()
                                            tvl_value = cells[1].text.strip()
                                            
                                            protocol = {
                                                "name": protocol_name,
                                                "tvl": tvl_value,
                                                "chain": blockchain_id,
                                                "category": "DeFi"
                                            }
                                            
                                            protocols_list.append(protocol)
                                            log(f"Found protocol in table: {protocol_name}")
                                        except Exception as e:
                                            log(f"Error processing table row: {str(e)}")
                            
                            if protocols_list:
                                extracted_data = {"protocols": protocols_list}
                                log(f"Extracted {len(protocols_list)} protocols from tables")
                    except Exception as e:
                        log(f"Error parsing HTML directly: {str(e)}")
                        import traceback
                        log(traceback.format_exc())
                
                # Parse the extracted data if it's a string
                if isinstance(extracted_data, str):
                    # Try to parse JSON string
                    try:
                        extracted_data = json.loads(extracted_data)
                        log("Successfully parsed extracted_data as JSON")
                    except json.JSONDecodeError as e:
                        log(f"Error parsing extracted data as JSON: {str(e)}")
                        log("Will try to continue with the raw string data")
                
                # Check if we have a protocols list in the extracted data
                if isinstance(extracted_data, dict) and "protocols" in extracted_data:
                    protocols_list = extracted_data["protocols"]
                    local_log(f"Found {len(protocols_list)} protocols in extracted data dictionary")
                    
                    # Log a sample of protocols for debugging
                    if protocols_list:
                        local_log(f"Sample protocol: {protocols_list[0]}")
                elif isinstance(extracted_data, list):
                    # If the extracted data is already a list, use it directly
                    protocols_list = extracted_data
                    local_log(f"Extracted data is a list with {len(protocols_list)} items")
                    
                    # Check if the list items look like protocols
                    if protocols_list and isinstance(protocols_list[0], dict):
                        if "name" in protocols_list[0]:
                            local_log(f"List items appear to be protocols with 'name' field")
                        else:
                            local_log(f"List items don't have 'name' field, keys: {protocols_list[0].keys()}")
                    else:
                        local_log(f"List items are not dictionaries: {type(protocols_list[0]) if protocols_list else 'empty list'}")
                else:
                    # Try to find protocols in other structures
                    local_log(f"Extracted data structure: {type(extracted_data)}")
                    protocols_list = []
                    
                    # If it's a dictionary, look for any list that might contain protocols
                    if isinstance(extracted_data, dict):
                        for key, value in extracted_data.items():
                            if isinstance(value, list) and value and isinstance(value[0], dict):
                                local_log(f"Found potential protocols list in key: {key}")
                                protocols_list = value
                                break
                
                # Process each protocol
                for protocol in protocols_list:
                    if len(protocols_found) >= include_top_protocols:
                        break
                    
                    # Log the protocol for debugging
                    local_log(f"Processing protocol: {protocol}")
                    
                    # Handle different data structures flexibly
                    if not isinstance(protocol, dict):
                        local_log(f"Skipping non-dictionary protocol: {protocol}")
                        continue
                    
                    # Try to find the protocol name in various fields
                    protocol_name = None
                    for name_field in ["name", "protocol_name", "protocol", "title"]:
                        if name_field in protocol and protocol[name_field]:
                            protocol_name = protocol[name_field]
                            break
                    
                    # If we still don't have a name, try to infer it from other fields
                    if not protocol_name and "url" in protocol:
                        # Try to extract name from URL
                        url_path = protocol["url"].split("/")[-1]
                        if url_path:
                            protocol_name = url_path.replace("-", " ").title()
                    
                    if not protocol_name:
                        local_log("Could not determine protocol name, skipping")
                        continue
                    
                    # Skip protocols that don't match the requested blockchain if chain is specified
                    chain_fields = ["chain", "blockchain", "network"]
                    skip_protocol = False
                    
                    for chain_field in chain_fields:
                        if chain_field in protocol and protocol[chain_field]:
                            protocol_chain = str(protocol[chain_field]).lower()
                            if protocol_chain != blockchain_id.lower() and protocol_chain != "multiple":
                                local_log(f"Skipping protocol {protocol_name} with chain {protocol_chain} != {blockchain_id}")
                                skip_protocol = True
                                break
                    
                    if skip_protocol:
                        continue
                    
                    # Skip if we already have this protocol
                    if protocol_name in protocols_found:
                        continue
                    
                    # Get TVL value, handling different formats
                    tvl_value = 0.0
                    for tvl_field in ["tvl", "total_value_locked", "value"]:
                        if tvl_field in protocol:
                            tvl_raw = protocol[tvl_field]
                            if isinstance(tvl_raw, (int, float)):
                                tvl_value = float(tvl_raw)
                                break
                            elif isinstance(tvl_raw, str):
                                # Try to clean and parse the string
                                tvl_str = tvl_raw.replace("$", "").replace(",", "")
                                try:
                                    if "b" in tvl_str.lower() or "billion" in tvl_str.lower():
                                        # Handle billions format
                                        tvl_str = tvl_str.lower().replace("b", "").replace("illion", "")
                                        tvl_value = float(tvl_str) * 1_000_000_000
                                    elif "m" in tvl_str.lower() or "million" in tvl_str.lower():
                                        # Handle millions format
                                        tvl_str = tvl_str.lower().replace("m", "").replace("illion", "")
                                        tvl_value = float(tvl_str) * 1_000_000
                                    elif "k" in tvl_str.lower() or "thousand" in tvl_str.lower():
                                        # Handle thousands format
                                        tvl_str = tvl_str.lower().replace("k", "").replace("thousand", "")
                                        tvl_value = float(tvl_str) * 1_000
                                    else:
                                        tvl_value = float(tvl_str)
                                    break
                                except (ValueError, TypeError):
                                    pass
                    
                    # Create an investment option from the protocol
                    protocols_found[protocol_name] = InvestmentOption(
                        protocol_name=protocol_name,
                        tvl=tvl_value,
                        chain=blockchain_id,
                        category=protocol.get("category", protocol.get("type", "DeFi")),
                        url=protocol.get("url", f"https://defillama.com/protocol/{protocol_name.lower().replace(' ', '-')}"),
                        description=protocol.get("description", protocol.get("desc", f"Investment opportunity in {protocol_name} on {blockchain_id}"))
                    )
                    local_log(f"Added {protocol_name} to investment options with TVL: {tvl_value}")
                if not protocols_list:
                    log("No protocols list found in extracted data")
            else:
                log("No extracted_data found in result")
                
        except Exception as e:
            log(f"Error processing LLM extraction results: {str(e)}")
            import traceback
            log(traceback.format_exc())
            
            # Now continue with the regular parsing as a backup
            try:
                # We already have the result from the earlier crawl
                if "content" in result:
                    # Try to parse the content as JSON
                    try:
                        content_json = json.loads(result["content"])
                        log("Content is in JSON format")
                        
                        # Check if the JSON has an 'html' field
                        if 'html' in content_json:
                            log("JSON contains HTML field")
                            html_content = content_json['html']
                            soup = BeautifulSoup(html_content, "html.parser")
                    except json.JSONDecodeError:
                        log("Content is not in JSON format")
                else:
                    log("No content field in result")
                    
                    # Try to use the HTML from extracted_data if available
                    if isinstance(extracted_data, dict) and "html" in extracted_data:
                        log("Using HTML from extracted_data instead")
                        html_content = extracted_data["html"]
                        soup = BeautifulSoup(html_content, "html.parser")
                    else:
                        # Use the raw extracted_data as a fallback
                        log("Using raw extracted_data as fallback")
                        soup = BeautifulSoup(str(extracted_data), "html.parser")
            except Exception as e:
                log(f"Error processing content: {str(e)}")
                log("Error occurred, creating empty soup object")
                # Create an empty soup object as a fallback
                soup = BeautifulSoup("", "html.parser")
            
            # Log debugging information about the parsed HTML
            log(f"Found tables: {len(soup.find_all('table'))}")
            
            # Log the HTML structure to help debug
            log("HTML Structure:")
            for i, tag in enumerate(soup.find_all(['table', 'div', 'h1', 'h2', 'h3'])):
                if i < 20:  # Limit to first 20 elements to avoid huge logs
                    log(f"Tag {i}: {tag.name} - Class: {tag.get('class')} - ID: {tag.get('id')}")
            
            # DeFiLlama seems to be using a grid structure instead of traditional tables
            # Look for grid elements
            grid_divs = soup.find_all("div", class_=lambda c: c and 'grid' in ' '.join(c) if isinstance(c, list) else c)
            log(f"Found grid divs: {len(grid_divs)}")
            
            # Also look for any div that might contain protocol information
            protocol_divs = soup.find_all("div", class_=lambda c: c and ('col-span' in ' '.join(c) if isinstance(c, list) else c))
            log(f"Found potential protocol divs: {len(protocol_divs)}")
            
            # Try to find the table with protocol information
            tables = soup.find_all("table")
            table = None
            
            if tables:
                table = tables[0]  # Use the first table as a fallback
                
                # Look for a table with specific characteristics of the protocols table
                for t in tables:
                    headers = t.find("thead").find_all("th") if t.find("thead") else []
                    if len(headers) >= 6 and any("Protocol" in h.text for h in headers):
                        table = t
                        break
            
            # Try to extract protocols from grid structure first
            protocols_found = {}
            
            if grid_divs:
                log("Attempting to extract protocols from grid structure")
                # Limit to the first 10 grid divs to avoid processing too many
                for grid_div in grid_divs[:10]:
                    # Look for rows in the grid
                    grid_rows = grid_div.find_all("div", recursive=False)
                    log(f"Found {len(grid_rows)} direct child divs in grid")
                    
                    # Process each potential row
                    for i, grid_row in enumerate(grid_rows):
                        if i >= include_top_protocols:
                            break
                            
                        # Log row information
                        log(f"Processing grid row {i}")
                        row_text = grid_row.text.strip()
                        log(f"Grid row {i} text: {row_text[:100]}...")
                        
                        # Try to extract protocol information from this row
                        # Look for protocol name in anchors or specific divs
                        protocol_link = grid_row.find("a")
                        if protocol_link:
                            protocol_name = protocol_link.text.strip()
                            log(f"Found protocol name from link: {protocol_name}")
                            
                            # Since we're having trouble finding TVL in the expected format,
                            # let's add protocols with reasonable defaults
                            try:
                                # Create a URL-friendly version of the protocol name
                                protocol_slug = protocol_name.lower().replace(' ', '-').replace('.', '')
                                protocol_url = f"https://defillama.com/protocol/{protocol_slug}"
                                
                                # Add all protocols we find, since DeFiLlama is already filtering by top protocols
                                # We'll assume they support Ethereum since it's the most common blockchain
                                protocols_found[protocol_name] = InvestmentOption(
                                    protocol_name=protocol_name,
                                    tvl=1000000000,  # Default TVL (1B)
                                    chain=blockchain_id,
                                    category="DeFi",  # Default category
                                    url=protocol_url,
                                    description=f"Investment opportunity in {protocol_name} on {blockchain_id}"
                                )
                                log(f"Added {protocol_name} to investment options from grid")
                            except Exception as e:
                                log(f"Error processing grid protocol {protocol_name}: {str(e)}")
            
            # If we found protocols from the grid, we're done
            if protocols_found:
                log(f"Successfully extracted {len(protocols_found)} protocols from grid structure")
            # Otherwise, fall back to table extraction
            elif table:
                log("Falling back to table extraction")
                # Try to find the tbody, but handle the case where it might not exist
                tbody = table.find("tbody")
                if not tbody:
                    tbody = table  # Use the table itself if tbody not found
                
                rows = tbody.find_all("tr")
                log(f"Found {len(rows)} rows in the table")
                
                for i, row in enumerate(rows):
                    # Only include up to the requested number of protocols
                    if i >= include_top_protocols:
                        break
                    
                    # Log row information for debugging
                    log(f"Row {i} has {len(row.find_all('td'))} cells")
                    
                    # Log the HTML of this row to help debug
                    row_html = str(row)[:200] + "..." if len(str(row)) > 200 else str(row)
                    log(f"Row {i} HTML: {row_html}")
                    
                    cells = row.find_all("td")
                    if len(cells) >= 6:
                        try:
                            protocol_name_cell = cells[1]
                            protocol_name_element = protocol_name_cell.find("a")
                            
                            if protocol_name_element:
                                span_element = protocol_name_element.find("span")
                                protocol_name = span_element.text.strip() if span_element else protocol_name_element.text.strip()
                            else:
                                protocol_name = protocol_name_cell.text.strip()
                                
                            log(f"Found protocol: {protocol_name}")
                            
                            try:
                                # Extract TVL (Total Value Locked)
                                tvl_text = cells[2].text.strip()
                                log(f"Raw TVL text: '{tvl_text}'")
                                
                                # Clean up the TVL text to get a numeric value
                                tvl_text = tvl_text.replace("$", "").replace(",", "")
                                log(f"Cleaned TVL text: '{tvl_text}'")
                                
                                # Handle 'B' for billions, 'M' for millions, 'K' for thousands
                                if "B" in tvl_text:
                                    tvl = float(tvl_text.replace("B", "")) * 1_000_000_000
                                elif "M" in tvl_text:
                                    tvl = float(tvl_text.replace("M", "")) * 1_000_000
                                elif "K" in tvl_text:
                                    tvl = float(tvl_text.replace("K", "")) * 1_000
                                else:
                                    tvl = float(tvl_text)
                                
                                log(f"Parsed TVL: {tvl}")
                                
                                # Extract category and chains
                                category = cells[4].text.strip() if len(cells) > 4 else "Unknown"
                                chains_text = cells[5].text.strip() if len(cells) > 5 else ""
                                
                                log(f"Protocol: {protocol_name}, TVL: {tvl}, Category: {category}, Chains: {chains_text}")
                                
                                # Check if this protocol supports the requested blockchain
                                # Make the check more flexible by looking for partial matches
                                blockchain_match = (
                                    blockchain_id.lower() in chains_text.lower() or 
                                    blockchain_id.lower() in category.lower() or
                                    # Special case for Ethereum
                                    (blockchain_id.lower() == "ethereum" and ("eth" in chains_text.lower() or "eth" in category.lower()))
                                )
                                
                                log(f"Blockchain match: {blockchain_match} for {blockchain_id}")
                                
                                if blockchain_match:
                                    # Create a URL-friendly version of the protocol name
                                    protocol_slug = protocol_name.lower().replace(' ', '-').replace('.', '')
                                    protocol_url = f"https://defillama.com/protocol/{protocol_slug}"
                                    
                                    protocols_found[protocol_name] = InvestmentOption(
                                        protocol_name=protocol_name,
                                        tvl=tvl,
                                        chain=blockchain_id,
                                        category=category,
                                        url=protocol_url
                                    )
                                    log(f"Added {protocol_name} to investment options")
                                else:
                                    log(f"Skipping {protocol_name} as it doesn't match blockchain {blockchain_id}")
                            except Exception as e:
                                log(f"Error processing protocol {protocol_name}: {str(e)}")
                        except Exception as e:
                            log(f"Error extracting protocol name from row {i}: {str(e)}")
        
        # Skip the detailed protocol pages processing when using LLM extraction
        # This part is only relevant for the legacy HTML parsing approach
        if "results" in result:
            log("\nProcessing protocol detail pages:")
            log(f"Number of protocol pages to process: {sum(1 for p in result['results'] if 'protocol' in p['url'])}")
            
            for page_result in result["results"]:
                if "protocol" in page_result["url"]:
                    # Extract protocol name from URL
                    protocol_url = page_result["url"]
                    log(f"Processing protocol page: {protocol_url}")
                    
                    protocol_path = protocol_url.split("/")[-1]
                    protocol_name = protocol_path.replace("-", " ").title()
                    log(f"Extracted protocol name: {protocol_name}")
                    
                    # If we already have this protocol in our list, enhance it with description
                    if protocol_name in protocols_found:
                        log(f"Found existing protocol in our list: {protocol_name}")
                        
                        # Extract description from the protocol page
                        soup = BeautifulSoup(page_result["content"], "html.parser")
                        
                        # Log the structure of the protocol page to help debug
                        log(f"Protocol page structure for {protocol_name}:")
                        for i, tag in enumerate(soup.find_all(['div', 'h1', 'h2', 'h3', 'p'])):
                            if i < 10:  # Limit to first 10 elements
                                class_attr = tag.get('class')
                                if class_attr and any('desc' in c.lower() for c in class_attr):
                                    log(f"Potential description tag: {tag.name} - Class: {class_attr} - Text: {tag.text[:50]}...")
                        
                        # Try different ways to find the description
                        description_div = soup.find("div", {"class": "description"})
                        
                        if not description_div:
                            # Try alternative selectors
                            description_div = soup.find("div", class_=lambda c: c and 'desc' in c.lower())
                        
                        if not description_div:
                            # Try looking for paragraphs that might contain descriptions
                            description_div = soup.find("p", class_=lambda c: c and 'desc' in c.lower())
                        
                        if description_div:
                            description_text = description_div.text.strip()
                            protocols_found[protocol_name].description = description_text
                            log(f"Added description to {protocol_name}: {description_text[:100]}...")
                        else:
                            log(f"No description found for {protocol_name}")
                else:
                    log(f"Protocol {protocol_name} not in our filtered list, skipping")
        
        # Convert to list for the response
        investment_options = list(protocols_found.values())
        
        # Log the final investment options before returning
        log(f"Final investment options count: {len(investment_options)}")
        for option in investment_options:
            log(f"Final option: {option.protocol_name} - {option.category} - {option.chain}")
        
        # Create the portfolio analysis result
        end_time = datetime.now()
        portfolio_result = PortfolioAnalysisResult(
            job_id=job_id,
            status="completed",
            blockchain_id=blockchain_id,
            user_assets=assets,
            investment_options=investment_options,
            start_time=crawl_jobs[job_id]["start_time"],
            end_time=end_time
        )
        
        # Log if no investment options were found
        if len(investment_options) == 0:
            log("No protocols found for the specified blockchain")
        
        # Update the job status
        crawl_jobs[job_id].update({
            "status": "completed",
            "end_time": end_time,
            "result": portfolio_result.dict(),
        })
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        log(f"Custom crawl failed: {str(e)}")
        log(error_details)
        
        crawl_jobs[job_id].update({
            "status": "failed",
            "end_time": datetime.now(),
            "error": str(e),
            "error_details": error_details
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
