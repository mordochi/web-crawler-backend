import asyncio
from typing import Dict, List, Optional, Any, Union
import os
import json
import re
from pathlib import Path
import logging

from crawl4ai import AsyncWebCrawler, CrawlResult
from crawl4ai import BFSDeepCrawlStrategy, DFSDeepCrawlStrategy, URLPatternFilter, FilterChain
from crawl4ai import JsonCssExtractionStrategy, LLMExtractionStrategy, LLMConfig
from llm_providers import get_llm_config, LLMProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCrawler:
    """
    A simple web crawler that crawls a single page and extracts its content.
    """
    
    async def crawl(
        self, 
        url: str, 
        output_format: str = "markdown", 
        wait_for_seconds: float = 0,
        capture_screenshot: bool = False,
        extract_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Crawl a single web page and extract its content.
        
        Args:
            url: The URL to crawl
            output_format: The output format (markdown, html, text, json)
            wait_for_seconds: Time to wait after page load (for dynamic content)
            capture_screenshot: Whether to capture a screenshot of the page
            extract_metadata: Whether to extract metadata from the page
            
        Returns:
            A dictionary containing the crawl results
        """
        async with AsyncWebCrawler() as crawler:
            # Configure crawler options
            crawler.wait_for_seconds = wait_for_seconds
            
            # Perform the crawl
            result = await crawler.arun(
                url=url,
                capture_screenshot=capture_screenshot,
                extract_metadata=extract_metadata
            )
            
            # Prepare result based on requested format
            response = {
                "url": url,
                "title": result.title,
                "links": result.links,
            }
            
            if output_format == "markdown":
                response["content"] = result.markdown
            elif output_format == "html":
                response["content"] = result.html
            elif output_format == "text":
                response["content"] = result.text
            elif output_format == "json":
                response["content"] = result.json
            
            # Include metadata if requested
            if extract_metadata and result.metadata:
                response["metadata"] = result.metadata
                
            # Include screenshot if captured
            if capture_screenshot and result.screenshot:
                # Save screenshot to a file
                screenshot_dir = Path("screenshots")
                screenshot_dir.mkdir(exist_ok=True)
                
                screenshot_filename = f"{url.replace('://', '_').replace('/', '_')}.png"
                screenshot_path = screenshot_dir / screenshot_filename
                
                with open(screenshot_path, "wb") as f:
                    f.write(result.screenshot)
                
                response["screenshot_path"] = str(screenshot_path)
            
            return response


class DeepCrawler:
    """
    A deep web crawler that crawls multiple pages starting from a root URL.
    """
    
    async def deep_crawl(
        self,
        url: str,
        strategy: str = "bfs",
        max_pages: int = 10,
        max_depth: int = 3,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        output_format: str = "markdown"
    ) -> Dict[str, Any]:
        """
        Perform a deep crawl starting from a root URL.
        
        Args:
            url: The root URL to start crawling from
            strategy: The crawling strategy ('bfs' or 'dfs')
            max_pages: Maximum number of pages to crawl
            max_depth: Maximum depth to crawl
            include_patterns: URL patterns to include (regex)
            exclude_patterns: URL patterns to exclude (regex)
            output_format: The output format (markdown, html, text, json)
            
        Returns:
            A dictionary containing the deep crawl results
        """
        async with AsyncWebCrawler() as crawler:
                        # Configure deep crawl strategy based on the specified strategy
            if strategy.lower() == "bfs":
                deep_crawl_strategy = BFSDeepCrawlStrategy(
                    max_depth=max_depth,
                    max_pages=max_pages
                )
            else:  # dfs strategy
                deep_crawl_strategy = DFSDeepCrawlStrategy(
                    max_depth=max_depth,
                    max_pages=max_pages
                )
            
            # For now, we'll skip URL filtering since we're having issues with the URLPatternFilter class
            # We'll implement a simple post-processing filter instead
            filter_chain = None
            
            # Since AsyncWebCrawler doesn't have a deep_crawl method, we'll use arun
            # and implement a custom solution for deep crawling
            
            # First, crawl the main page
            result = await crawler.arun(url=url)
            
            # Create a list to store all results
            all_results = [result]
            visited_urls = {url}
            
            # Queue for BFS or stack for DFS
            url_queue = [(url, 1)]  # (url, depth)
            
            # Process URLs according to the strategy
            while url_queue and len(all_results) < max_pages:
                current_url, current_depth = url_queue.pop(0 if strategy.lower() == "bfs" else -1)
                
                # Skip if we've reached max depth
                if current_depth >= max_depth:
                    continue
                
                # Get links from the current page
                current_result = next((r for r in all_results if r.url == current_url), None)
                if not current_result or not current_result.links:
                    continue
                
                # Process links
                for link in current_result.links:
                    # Check if link is a string or a dictionary
                    if isinstance(link, dict):
                        link_url = link.get("url", "")
                    else:
                        # Assume it's a string
                        link_url = link
                    
                    # Skip if empty or already visited
                    if not link_url or link_url in visited_urls:
                        continue
                    
                    # Apply include/exclude patterns if provided
                    if include_patterns and not any(re.search(pattern, link_url) for pattern in include_patterns):
                        continue
                    
                    if exclude_patterns and any(re.search(pattern, link_url) for pattern in exclude_patterns):
                        continue
                    
                    # Crawl the link
                    try:
                        link_result = await crawler.arun(url=link_url)
                        link_result.depth = current_depth + 1  # Add depth information
                        
                        all_results.append(link_result)
                        visited_urls.add(link_url)
                        
                        # Add to queue/stack for further processing
                        url_queue.append((link_url, current_depth + 1))
                        
                        # Break if we've reached max pages
                        if len(all_results) >= max_pages:
                            break
                    except Exception as e:
                        print(f"Error crawling {link_url}: {str(e)}")
            
            # Use all_results as our results
            results = all_results
            
            # Process results
            processed_results = []
            for result in results:
                # Get the available attributes safely
                page_result = {"url": getattr(result, "url", "")}
                
                # Add depth if available (we added this attribute ourselves)
                if hasattr(result, "depth"):
                    page_result["depth"] = result.depth
                else:
                    page_result["depth"] = 0  # Default depth
                
                # Add links if available
                if hasattr(result, "links"):
                    page_result["links"] = result.links
                
                # Try to get content based on the requested format
                if output_format == "markdown" and hasattr(result, "markdown"):
                    page_result["content"] = result.markdown
                elif output_format == "html" and hasattr(result, "html"):
                    page_result["content"] = result.html
                elif output_format == "text" and hasattr(result, "text"):
                    page_result["content"] = result.text
                elif output_format == "json" and hasattr(result, "json"):
                    # Check if json is a method or a property
                    if callable(result.json):
                        try:
                            page_result["content"] = result.json()
                        except Exception as e:
                            print(f"Error calling json() method: {str(e)}")
                            # Fallback to html or text
                            if hasattr(result, "html"):
                                page_result["content"] = result.html
                            elif hasattr(result, "text"):
                                page_result["content"] = result.text
                    else:
                        page_result["content"] = result.json
                else:
                    # Fallback to text content if available
                    for attr in ["text", "html", "markdown"]:
                        if hasattr(result, attr):
                            page_result["content"] = getattr(result, attr)
                            break
                
                processed_results.append(page_result)
            
            return {
                "root_url": url,
                "pages_crawled": len(processed_results),
                "strategy": strategy,
                "max_depth": max_depth,
                "results": processed_results
            }


class CustomCrawler:
    """
    A custom crawler that supports specialized extraction based on queries or selectors.
    """
    
    async def custom_crawl(
        self,
        url: str,
        query: Optional[str] = None,
        selectors: Optional[Dict[str, str]] = None,
        extraction_strategy: str = "heuristic",
        output_format: str = "markdown",
        llm_extraction_strategy: Optional[LLMExtractionStrategy] = None
    ) -> Dict[str, Any]:
        """
        Perform a custom crawl with specialized extraction.
        
        Args:
            url: The URL to crawl
            query: A natural language query to guide extraction
            selectors: CSS selectors for structured extraction
            extraction_strategy: Strategy for extraction ('heuristic', 'llm', or 'selectors')
            output_format: The output format (markdown, html, text, json)
            llm_extraction_strategy: Optional LLMExtractionStrategy instance for advanced LLM extraction
            
        Returns:
            A dictionary containing the custom crawl results
        """
        async with AsyncWebCrawler() as crawler:
            # Configure extraction based on strategy
            if extraction_strategy == "llm":
                if llm_extraction_strategy:
                    # Use the provided LLM extraction strategy
                    result = await crawler.arun(
                        url=url,
                        extraction_strategy=llm_extraction_strategy
                    )
                    
                    # Create a safe response with error handling
                    response = {
                        "url": url,
                        "extraction_strategy": "llm",
                        "extracted_data": getattr(result, 'json', {})
                    }
                    
                    # Add optional fields if they exist
                    if hasattr(result, 'title'):
                        response["title"] = result.title
                    if hasattr(result, 'links'):
                        response["links"] = result.links
                        
                    return response
                elif query:
                    # Use LLM-guided extraction with query if no strategy provided
                    result = await crawler.arun(
                        url=url,
                        question=query
                    )
                    
                    # Create a safe response with error handling
                    response = {
                        "url": url,
                        "extraction_strategy": "llm",
                        "query": query
                    }
                    
                    # Add optional fields if they exist
                    if hasattr(result, 'title'):
                        response["title"] = result.title
                    if hasattr(result, 'links'):
                        response["links"] = result.links
                    
                    # Add content based on the requested output format
                    if output_format == "markdown" and hasattr(result, 'markdown'):
                        response["content"] = result.markdown
                    elif output_format == "text" and hasattr(result, 'text'):
                        response["content"] = result.text
                    elif output_format == "html" and hasattr(result, 'html'):
                        response["content"] = result.html
                    elif output_format == "json" and hasattr(result, 'json'):
                        response["content"] = result.json
                    else:
                        # Fallback to any available content
                        for attr in ['text', 'markdown', 'html', 'json']:
                            if hasattr(result, attr):
                                response["content"] = getattr(result, attr)
                                break
                        
                    return response
            
            elif selectors:
                # Use CSS selectors for structured extraction
                extraction_strategy = JsonCssExtractionStrategy(selectors=selectors)
                
                result = await crawler.arun(
                    url=url,
                    extraction_strategy=extraction_strategy
                )
                
                return {
                    "url": url,
                    "extraction_strategy": "selectors",
                    "selectors": selectors,
                    "title": result.title,
                    "extracted_data": result.json,
                    "links": result.links
                }
            
            elif extraction_strategy == "html":
                # Use HTML extraction for direct parsing
                result = await crawler.arun(
                    url=url
                )
                
                # Create a safe response with error handling
                response = {
                    "url": url,
                    "extraction_strategy": "html",
                    "extracted_data": {
                        "url": url,
                        "html": result.html if hasattr(result, 'html') else ""
                    }
                }
                
                # Add optional fields if they exist
                if hasattr(result, 'title'):
                    response["title"] = result.title
                if hasattr(result, 'links'):
                    response["links"] = result.links
                
                return response
            
            else:
                # Fall back to heuristic extraction
                result = await crawler.arun(
                    url=url
                )
                
                response = {
                    "url": url,
                    "extraction_strategy": "heuristic"
                }
                
                # Add optional fields if they exist
                if hasattr(result, 'title'):
                    response["title"] = result.title
                if hasattr(result, 'links'):
                    response["links"] = result.links
                
                if output_format == "markdown" and hasattr(result, 'markdown'):
                    response["content"] = result.markdown
                elif output_format == "html" and hasattr(result, 'html'):
                    response["content"] = result.html
                elif output_format == "text" and hasattr(result, 'text'):
                    response["content"] = result.text
                elif output_format == "json" and hasattr(result, 'json'):
                    response["content"] = result.json
                else:
                    # Fallback to any available content
                    for attr in ['text', 'markdown', 'html', 'json']:
                        if hasattr(result, attr):
                            response["content"] = getattr(result, attr)
                            break
                
                return response
