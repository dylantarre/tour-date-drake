# Tour Date Parser

A service that extracts and formats tour dates from images or text using various vision models through OpenRouter.

## Setup

1. Copy `.env.example` to `.env` and add your OpenRouter API key:
```bash
cp .env.example .env
```

2. Run with Docker Compose:
```bash
docker-compose up -d
```

The service will be available at http://localhost:4242

## Development

To run in development mode with live reload:
```bash
docker-compose up
```

To rebuild after making changes:
```bash
docker-compose up --build
```

## API Usage

### Endpoint: POST /parse_tour_dates

Request body:
```json
{
    "image_url": "https://example.com/tour-poster.jpg",  // Optional
    "text_input": "Tour dates in text format",           // Optional
    "output_format": "lambgoat",                         // or "needledrop"
    "model": "openai/gpt-4-vision-preview"               // Optional, defaults to OpenAI
}
```

Example formats:

1. Lambgoat:
```
MM/DD/YYYY - Venue Name - City, State/Country
```

2. The Needle Drop:
```
Date: MM/DD/YYYY
Venue: Venue Name
Location: City, State/Country
```

### Example cURL:
```bash
curl -X POST "http://localhost:4242/parse_tour_dates" \
     -H "Content-Type: application/json" \
     -d '{
           "image_url": "https://example.com/tour-poster.jpg",
           "output_format": "lambgoat",
           "model": "openai/gpt-4-vision-preview"
         }'
``` 