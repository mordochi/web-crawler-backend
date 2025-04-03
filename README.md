# Web Crawler Backend

A Python backend project that uses the crawl4ai library to crawl web pages and extract structured data.

## Features

- Web page crawling with markdown output
- Advanced filtering and content extraction
- API endpoints for crawling and data retrieval
- Customizable crawling strategies

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run post-installation setup for crawl4ai:
   ```
   crawl4ai-setup
   ```
4. Verify the installation:
   ```
   crawl4ai-doctor
   ```

## Usage

1. Start the backend server:
   ```
   uvicorn main:app --reload
   ```
2. Access the API documentation at `http://localhost:8000/docs`
3. Use the API endpoints to crawl websites and retrieve data

## API Endpoints

- `POST /api/crawl`: Crawl a website and return the results
- `POST /api/deep-crawl`: Perform a deep crawl with customizable options
- `GET /api/crawl-status/{job_id}`: Check the status of a crawling job

## Configuration

You can configure the crawler by modifying the `.env` file or environment variables.

## License

MIT
