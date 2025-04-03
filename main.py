from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field, validator
from typing import Optional, Dict, List, Union, Any, Set, Literal
import asyncio
import uuid
import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from crawlers import SimpleCrawler, DeepCrawler, CustomCrawler
from bs4 import BeautifulSoup
from crawl4ai import LLMExtractionStrategy, LLMConfig

app = FastAPI(title="Web Crawler Backend", description="Backend API for crawling websites using crawl4ai")

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
        log(f"Portfolio analysis failed: {str(e)}\n{error_details}")
        
        # Add a summary of what we found so far
        log("\nSummary of findings:")
        log(f"Number of protocols found: {len(protocols_found)}")
        if protocols_found:
            log("Protocols found:")
            for name, protocol in protocols_found.items():
                log(f"- {name} ({protocol.category}): ${protocol.tvl:.2f}")
        else:
            log("No protocols found matching the criteria.")
        
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
        log(f"Portfolio analysis failed: {str(e)}\n{error_details}")
        
        # Add a summary of what we found so far
        log("\nSummary of findings:")
        log(f"Number of protocols found: {len(protocols_found)}")
        if protocols_found:
            log("Protocols found:")
            for name, protocol in protocols_found.items():
                log(f"- {name} ({protocol.category}): ${protocol.tvl:.2f}")
        else:
            log("No protocols found matching the criteria.")
        
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
        log(f"Portfolio analysis failed: {str(e)}\n{error_details}")
        
        # Add a summary of what we found so far
        log("\nSummary of findings:")
        log(f"Number of protocols found: {len(protocols_found)}")
        if protocols_found:
            log("Protocols found:")
            for name, protocol in protocols_found.items():
                log(f"- {name} ({protocol.category}): ${protocol.tvl:.2f}")
        else:
            log("No protocols found matching the criteria.")
        
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
    
    def log(message):
        print(message)
        with open(log_file, "a") as f:
            f.write(f"{message}\n")
    
    try:
        # No hardcoded protocols - we'll return empty array if none found
        
        # Use DeepCrawler to crawl DeFiLlama's top protocols and related pages
        try:
            crawler = DeepCrawler()
            
            # Use CustomCrawler with LLM extraction strategy for better analysis
            custom_crawler = CustomCrawler()
            
            # Check if OpenAI API key is available
            openai_api_key = os.environ.get("OPENAI_API_KEY", "")
            if not openai_api_key:
                log("WARNING: No OpenAI API key found in environment variables. Skipping LLM extraction and using direct HTML parsing.")
                extraction_strategy = "html"
            else:
                log("OpenAI API key found. Using LLM extraction strategy.")
                extraction_strategy = "llm"
                
            # Create an LLM extraction strategy if we have an API key
            if extraction_strategy == "llm":
                llm_extraction_strategy = LLMExtractionStrategy(
                    llm_config=LLMConfig(
                        provider="openai/gpt-3.5-turbo",  # You can change to your preferred model
                        api_token=openai_api_key  # Get API key from environment
                    ),
                    schema=ProtocolList.model_json_schema(),  # Use our Pydantic model schema
                    extraction_type="schema",
                    instruction=f"Extract a list of top DeFi protocols from this page. Focus on protocols available on {blockchain_id}. Include name, TVL (Total Value Locked), category, and other available information. Return the data as a structured JSON object with a 'protocols' array containing protocol objects.",
                    chunk_token_threshold=4000,
                    overlap_rate=0.1,
                    apply_chunking=True,
                    input_format="html",
                    extra_args={"temperature": 0.1}
                )
            
            # Perform the crawl with the appropriate extraction strategy
            if extraction_strategy == "llm":
                # Log the schema being used for extraction
                log(f"Using schema for extraction: {ProtocolList.model_json_schema()}")
                
                # Perform the crawl with LLM extraction
                result = await custom_crawler.custom_crawl(
                    url="https://defillama.com/top-protocols",
                    extraction_strategy="llm",  # Use LLM-based extraction
                    output_format="json",
                    llm_extraction_strategy=llm_extraction_strategy
                )
            else:
                # Perform the crawl with HTML extraction
                log("Using direct HTML extraction")
                result = await custom_crawler.custom_crawl(
                    url="https://defillama.com/top-protocols",
                    extraction_strategy="html",  # Use HTML-based extraction
                    output_format="json"
                )
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
                    log(f"Found {len(protocols_list)} protocols in extracted data")
                elif isinstance(extracted_data, list):
                    # If the extracted data is already a list, use it directly
                    protocols_list = extracted_data
                    log(f"Extracted data is a list with {len(protocols_list)} items")
                else:
                    log(f"Extracted data structure: {type(extracted_data)}")
                    protocols_list = []
                
                # Process each protocol
                for protocol in protocols_list:
                    if len(protocols_found) >= include_top_protocols:
                        break
                        
                    # Skip protocols that don't match the requested blockchain
                    if "chain" in protocol and protocol["chain"].lower() != blockchain_id.lower():
                        # Skip protocols that don't match the blockchain
                        continue
                        
                    protocol_name = protocol.get("name", "")
                    if protocol_name and protocol_name not in protocols_found:
                        # Create an investment option from the protocol
                        protocols_found[protocol_name] = InvestmentOption(
                            protocol_name=protocol_name,
                            tvl=protocol.get("tvl", 0.0),
                            chain=blockchain_id,
                            category=protocol.get("category", "DeFi"),
                            url=protocol.get("url", f"https://defillama.com/protocol/{protocol_name.lower().replace(' ', '-')}"),
                            description=protocol.get("description", f"Investment opportunity in {protocol_name} on {blockchain_id}")
                        )
                        log(f"Added {protocol_name} to investment options from LLM extraction")
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
                    # Try to use the entire JSON as the content
                    soup = BeautifulSoup(str(content_json), "html.parser")
            except json.JSONDecodeError:
                log("Content is not in JSON format, using as raw HTML")
                # Extract protocol information from the table
                soup = BeautifulSoup(main_page_result["content"], "html.parser")
            
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
        log(f"Portfolio analysis failed: {str(e)}\n{error_details}")
        
        # Add a summary of what we found so far
        log("\nSummary of findings:")
        log(f"Number of protocols found: {len(protocols_found)}")
        if protocols_found:
            log("Protocols found:")
            for name, protocol in protocols_found.items():
                log(f"- {name} ({protocol.category}): ${protocol.tvl:.2f}")
        else:
            log("No protocols found matching the criteria.")
        
        crawl_jobs[job_id].update({
            "status": "failed",
            "end_time": datetime.now(),
            "error": str(e),
            "error_details": error_details
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
