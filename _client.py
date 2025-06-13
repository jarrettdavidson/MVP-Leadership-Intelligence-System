"""
Limitless API Client for MVP-Leadership Intelligence System

This module handles communication with the Limitless API to fetch lifelog data.
"""

import requests
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta


def get_lifelogs(api_key: str, date: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch lifelogs from Limitless API for a specific date.
    
    Args:
        api_key: Your Limitless API key
        date: Date in YYYY-MM-DD format
        limit: Maximum number of lifelogs to fetch
        
    Returns:
        List of lifelog dictionaries containing conversation data
        
    Raises:
        Exception: If API request fails or returns invalid data
    """
    
    # Limitless API endpoint - you may need to adjust this URL
    # Check your Limitless documentation for the correct endpoint
    base_url = "https://api.limitless.ai"  # Replace with actual Limitless API URL
    endpoint = f"{base_url}/lifelogs"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    params = {
        "date": date,
        "limit": limit,
        "include_content": True  # Ensure we get the actual conversation content
    }
    
    try:
        print(f"🔄 Fetching lifelogs from Limitless API for date: {date}")
        
        response = requests.get(endpoint, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Handle different possible response formats
        if isinstance(data, dict):
            if "lifelogs" in data:
                lifelogs = data["lifelogs"]
            elif "data" in data:
                lifelogs = data["data"]
            elif "results" in data:
                lifelogs = data["results"]
            else:
                # Assume the entire response is the lifelog data
                lifelogs = [data] if data else []
        elif isinstance(data, list):
            lifelogs = data
        else:
            raise ValueError(f"Unexpected response format: {type(data)}")
        
        print(f"✅ Successfully fetched {len(lifelogs)} lifelogs")
        
        # Validate that we have the expected data structure
        if lifelogs and len(lifelogs) > 0:
            first_log = lifelogs[0]
            
            # Check if we have content in any of the expected fields
            content_fields = ['contents', 'transcript', 'text', 'content', 'speech', 'audio_transcript']
            has_content = any(field in first_log and first_log[field] for field in content_fields)
            
            if not has_content:
                print("⚠️ Warning: Lifelogs found but no conversation content detected")
                print(f"Available fields in first lifelog: {list(first_log.keys())}")
        
        return lifelogs
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to Limitless API: {e}")
        raise Exception(f"Failed to fetch lifelogs from Limitless API: {e}")
    
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing Limitless API response: {e}")
        raise Exception(f"Invalid JSON response from Limitless API: {e}")
    
    except Exception as e:
        print(f"❌ Unexpected error fetching lifelogs: {e}")
        raise


def test_limitless_connection(api_key: str) -> bool:
    """
    Test the connection to Limitless API to verify credentials and endpoint.
    
    Args:
        api_key: Your Limitless API key
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Try to fetch a small amount of recent data to test connection
        yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
        test_logs = get_lifelogs(api_key, yesterday, limit=1)
        
        print(f"✅ Limitless API connection test successful")
        print(f"   Found {len(test_logs)} lifelogs for {yesterday}")
        
        if test_logs:
            print(f"   Sample lifelog keys: {list(test_logs[0].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Limitless API connection test failed: {e}")
        return False


def get_available_dates(api_key: str, days_back: int = 7) -> List[str]:
    """
    Get list of dates that have lifelog data available.
    
    Args:
        api_key: Your Limitless API key
        days_back: Number of days back to check
        
    Returns:
        List of dates (YYYY-MM-DD format) that have data
    """
    available_dates = []
    
    for i in range(days_back):
        check_date = (datetime.now() - timedelta(days=i)).date().isoformat()
        
        try:
            logs = get_lifelogs(api_key, check_date, limit=1)
            if logs:
                available_dates.append(check_date)
        except:
            # Skip dates that cause errors
            continue
    
    return available_dates


# Alternative implementation for different Limitless API structures
def get_lifelogs_alternative(api_key: str, date: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Alternative implementation if the main get_lifelogs function doesn't work.
    
    This version handles different possible API structures or local file formats.
    """
    
    # If you have a different API endpoint or local file structure,
    # implement the logic here
    
    # Example for local file-based approach:
    # try:
    #     import os
    #     lifelog_file = f"lifelogs_{date}.json"
    #     if os.path.exists(lifelog_file):
    #         with open(lifelog_file, 'r') as f:
    #             return json.load(f)
    # except Exception as e:
    #     print(f"Error reading local lifelog file: {e}")
    
    # Fallback to main implementation
    return get_lifelogs(api_key, date, limit)


if __name__ == "__main__":
    """
    Test script to verify Limitless API connection.
    Run this file directly to test your API connection.
    """
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("LIMITLESS_API_KEY")
    if not api_key:
        print("❌ LIMITLESS_API_KEY not found in environment variables")
        print("Please add your Limitless API key to the .env file")
        exit(1)
    
    print("Testing Limitless API connection...")
    
    # Test connection
    if test_limitless_connection(api_key):
        print("\n✅ Connection test passed!")
        
        # Show available dates
        print("\nChecking available dates...")
        dates = get_available_dates(api_key, days_back=5)
        print(f"Available dates with data: {dates}")
        
    else:
        print("\n❌ Connection test failed!")
        print("\nTroubleshooting steps:")
        print("1. Verify your LIMITLESS_API_KEY in .env file")
        print("2. Check if the API endpoint URL is correct")
        print("3. Ensure your Limitless app is recording conversations")
        print("4. Contact Limitless support for API documentation")
