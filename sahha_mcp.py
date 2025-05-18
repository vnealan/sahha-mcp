#!/usr/bin/env python3
"""
Sahha API MCP Client

This script provides a Model Context Protocol (MCP) compliant interface
to interact with the Sahha Platform v1 API.
"""

from typing import Any, Dict, List, Optional
import httpx
import json
from datetime import datetime
import logging
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sahha_mcp_client")

# Initialize FastMCP server
mcp = FastMCP("sahha")

# Constants
SAHHA_API_BASE = "https://api.sahha.ai"
USER_AGENT = "sahha-mcp-client/1.0"

# Global variables to store authentication tokens
ACCOUNT_TOKEN = None
PROFILE_TOKEN = None

# Helper functions for validation
def validate_date_format(date_str: Optional[str]) -> bool:
    """
    Validate that a string is in ISO date format.
    
    Args:
        date_str: Date string to validate
        
    Returns:
        True if valid or None, False otherwise
    """
    if not date_str:
        return True
    
    try:
        # Convert Z timezone marker to +00:00 for compatibility with fromisoformat
        datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return True
    except ValueError:
        return False


def validate_not_empty(value: Optional[str], param_name: str) -> Optional[str]:
    """
    Validate that a parameter is not empty if provided.
    
    Args:
        value: Parameter value
        param_name: Parameter name for error message
        
    Returns:
        Error message if invalid, None if valid
    """
    if value is not None and value.strip() == "":
        return f"Parameter '{param_name}' cannot be empty"
    return None


# Helper functions for formatting responses
def format_biomarker_response(response: List[Dict], limit_per_category: int = 5) -> List[str]:
    """
    Format biomarker response into readable text.
    
    Args:
        response: API response containing biomarkers
        limit_per_category: Maximum number of biomarkers to show per category
        
    Returns:
        List of formatted strings
    """
    if not response or len(response) == 0:
        return ["No biomarkers found with the specified criteria"]
    
    result = [f"Found {len(response)} biomarkers:"]
    
    # Group biomarkers by category
    grouped = {}
    for biomarker in response:
        category = biomarker.get("category", "Unknown")
        if category not in grouped:
            grouped[category] = []
        grouped[category].append(biomarker)
    
    for category, biomarkers in grouped.items():
        result.append(f"\n=== Category: {category} ===")
        
        for biomarker in biomarkers[:limit_per_category]:
            result.append(f"\n- Type: {biomarker.get('type', 'N/A')}")
            result.append(f"  Value: {biomarker.get('value', 'N/A')} {biomarker.get('unit', '')}")
            
            if "startDateTime" in biomarker and biomarker["startDateTime"]:
                result.append(f"  Period: {biomarker.get('startDateTime')} to {biomarker.get('endDateTime', 'N/A')}")
            
            if "periodicity" in biomarker and biomarker["periodicity"]:
                result.append(f"  Periodicity: {biomarker.get('periodicity')}")
        
        if len(biomarkers) > limit_per_category:
            result.append(f"  ... and {len(biomarkers) - limit_per_category} more {category} biomarkers")
    
    return result


def format_trend_response(response: List[Dict]) -> List[str]:
    """
    Format trend response into readable text.
    
    Args:
        response: API response containing trends
        
    Returns:
        List of formatted strings
    """
    if not response or len(response) == 0:
        return ["No trends found with the specified criteria"]
    
    result = [f"Found {len(response)} trends:"]
    
    for i, trend in enumerate(response, 1):
        result.append(f"\n--- Trend {i} ---")
        result.append(f"Name: {trend.get('name', 'N/A')}")
        result.append(f"Category: {trend.get('category', 'N/A')}")
        result.append(f"State: {trend.get('state', 'N/A')}")
        result.append(f"Unit: {trend.get('unit', 'N/A')}")
        result.append(f"Periodicity: {trend.get('periodicity', 'N/A')}")
        
        if "isHigherBetter" in trend and trend["isHigherBetter"] is not None:
            higher_better = "Yes" if trend["isHigherBetter"] else "No"
            result.append(f"Is Higher Better: {higher_better}")
        
        if "trendStartDateTime" in trend and trend["trendStartDateTime"]:
            result.append(f"Period: {trend.get('trendStartDateTime')} to {trend.get('trendEndDateTime', 'N/A')}")
        
        if "data" in trend and trend["data"]:
            data_points = len(trend["data"])
            result.append(f"Data Points: {data_points}")
            
            # Include some data point details
            if data_points > 0:
                result.append("\nData Point Samples:")
                for j, point in enumerate(trend["data"][:3], 1):  # Show up to 3 data points
                    result.append(f"  {j}. Value: {point.get('value', 'N/A')}")
                    if "startDateTime" in point and point["startDateTime"]:
                        result.append(f"     Period: {point.get('startDateTime')} to {point.get('endDateTime', 'N/A')}")
                    if "percentChangeFromPrevious" in point and point["percentChangeFromPrevious"] is not None:
                        percent_change = point["percentChangeFromPrevious"] * 100
                        direction = "increase" if percent_change >= 0 else "decrease"
                        result.append(f"     Change: {abs(percent_change):.1f}% {direction} from previous")
    
    return result


def format_comparison_response(response: List[Dict]) -> List[str]:
    """
    Format comparison response into readable text.
    
    Args:
        response: API response containing comparisons
        
    Returns:
        List of formatted strings
    """
    if not response or len(response) == 0:
        return ["No comparisons found with the specified criteria"]
    
    result = [f"Found {len(response)} comparisons:"]
    
    for i, comparison in enumerate(response, 1):
        result.append(f"\n--- Comparison {i} ---")
        result.append(f"Name: {comparison.get('name', 'N/A')}")
        result.append(f"Category: {comparison.get('category', 'N/A')}")
        result.append(f"Value: {comparison.get('value', 'N/A')} {comparison.get('unit', '')}")
        result.append(f"Periodicity: {comparison.get('periodicity', 'N/A')}")
        
        if "isHigherBetter" in comparison and comparison["isHigherBetter"] is not None:
            higher_better = "Yes" if comparison["isHigherBetter"] else "No"
            result.append(f"Is Higher Better: {higher_better}")
        
        if "startDateTime" in comparison and comparison["startDateTime"]:
            result.append(f"Period: {comparison.get('startDateTime')} to {comparison.get('endDateTime', 'N/A')}")
        
        if "data" in comparison and comparison["data"]:
            data_points = len(comparison["data"])
            result.append(f"Comparison Points: {data_points}")
            
            # Include some comparison data point details
            if data_points > 0:
                result.append("\nComparison Data:")
                for j, point in enumerate(comparison["data"][:3], 1):  # Show up to 3 comparison points
                    result.append(f"  {j}. Type: {point.get('type', 'N/A')}")
                    result.append(f"     Value: {point.get('value', 'N/A')}")
                    
                    if "percentile" in point and point["percentile"] is not None:
                        result.append(f"     Percentile: {point.get('percentile')}")
                    
                    if "difference" in point and point["difference"] is not None:
                        result.append(f"     Difference: {point.get('difference')}")
                    
                    if "percentageDifference" in point and point["percentageDifference"] is not None:
                        percent_diff = point["percentageDifference"] * 100
                        result.append(f"     Percentage Difference: {percent_diff:.1f}%")
                    
                    if "state" in point and point["state"]:
                        result.append(f"     State: {point.get('state')}")
    
    return result


def format_integration_response(response: List[Dict]) -> List[str]:
    """
    Format integration response into readable text.
    
    Args:
        response: API response containing integrations
        
    Returns:
        List of formatted strings
    """
    if not response or len(response) == 0:
        return ["No integrations found"]
    
    result = [f"Found {len(response)} integrations:"]
    
    for i, integration in enumerate(response, 1):
        result.append(f"\n--- Integration {i} ---")
        result.append(f"Type: {integration.get('integrationType', 'N/A')}")
        result.append(f"Identifier: {integration.get('integrationIdentifier', 'N/A')}")
        
        if "integrationValues" in integration and integration["integrationValues"]:
            result.append("Values:")
            for key, value in integration["integrationValues"].items():
                result.append(f"  {key}: {value}")
    
    return result


def format_archetype_response(response: List[Dict], limit: int = 10) -> List[str]:
    """
    Format archetype response into readable text.
    
    Args:
        response: API response containing archetypes
        limit: Maximum number of archetypes to show in detail
        
    Returns:
        List of formatted strings
    """
    if not response or len(response) == 0:
        return ["No archetypes found with the specified criteria"]
    
    result = [f"Found {len(response)} archetypes:"]
    
    for i, archetype in enumerate(response[:limit], 1):  # Limit detailed display
        result.append(f"\n--- Archetype {i} ---")
        result.append(f"Name: {archetype.get('name', 'N/A')}")
        result.append(f"Value: {archetype.get('value', 'N/A')}")
        result.append(f"Data Type: {archetype.get('dataType', 'N/A')}")
        result.append(f"Periodicity: {archetype.get('periodicity', 'N/A')}")
        
        if "startDateTime" in archetype and archetype["startDateTime"]:
            result.append(f"Period: {archetype.get('startDateTime')} to {archetype.get('endDateTime', 'N/A')}")
    
    if len(response) > limit:
        result.append(f"\n... and {len(response) - limit} more archetypes")
    
    return result


async def make_api_request(
    method: str,
    endpoint: str,
    auth_type: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Make a request to the Sahha API with proper error handling.
    
    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint path
        auth_type: Authentication type ('account', 'profile', or None)
        params: Query parameters
        data: Request body data
        headers: Additional headers
        
    Returns:
        API response as a dictionary
    """
    url = f"{SAHHA_API_BASE}{endpoint}"
    
    # Initialize headers
    if headers is None:
        headers = {}
    
    headers["User-Agent"] = USER_AGENT
    
    # Add authentication if specified
    if auth_type == 'account' and ACCOUNT_TOKEN:
        headers['Authorization'] = f"account {ACCOUNT_TOKEN}"
    elif auth_type == 'profile' and PROFILE_TOKEN:
        headers['Authorization'] = f"profile {PROFILE_TOKEN}"
    
    # Add content type for POST/PUT requests
    if method in ['POST', 'PUT', 'PATCH'] and 'Content-Type' not in headers:
        headers['Content-Type'] = 'application/json'
    
    # Make the request
    async with httpx.AsyncClient() as client:
        try:
            if method == "GET":
                response = await client.get(url, headers=headers, params=params, timeout=30.0)
            elif method == "POST":
                response = await client.post(url, headers=headers, params=params, json=data, timeout=30.0)
            elif method == "PUT":
                response = await client.put(url, headers=headers, params=params, json=data, timeout=30.0)
            elif method == "PATCH":
                response = await client.patch(url, headers=headers, params=params, json=data, timeout=30.0)
            elif method == "DELETE":
                response = await client.delete(url, headers=headers, params=params, json=data, timeout=30.0)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            # Return JSON response if available, otherwise return status code
            if response.text:
                return response.json()
            else:
                return {"status_code": response.status_code}
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return {
                "error": True,
                "status_code": e.response.status_code,
                "message": e.response.text
            }
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            return {"error": True, "message": str(e)}


@mcp.tool()
async def authenticate_account(client_id: str, client_secret: str) -> str:
    """
    Authenticate with the Sahha API using client credentials to get an account token.
    
    Args:
        client_id: Client ID for Sahha API
        client_secret: Client secret for Sahha API
        
    Returns:
        Status message indicating success or failure
    """
    global ACCOUNT_TOKEN
    
    # Validate input parameters
    if error_msg := validate_not_empty(client_id, "client_id"):
        return error_msg
    
    if error_msg := validate_not_empty(client_secret, "client_secret"):
        return error_msg
    
    data = {
        "clientId": client_id,
        "clientSecret": client_secret
    }
    
    response = await make_api_request(
        method="POST",
        endpoint="/api/v1/oauth/account/token",
        data=data
    )
    
    if "error" in response and response["error"]:
        return f"Authentication failed: {response.get('message', 'Unknown error')}"
    
    if "accountToken" in response:
        ACCOUNT_TOKEN = response["accountToken"]
        return "Successfully authenticated with account token"
    else:
        return f"Failed to authenticate: {json.dumps(response)}"


@mcp.tool()
async def get_profile_token(external_id: str, read_only: bool = False, lifetime: int = 86400) -> str:
    """
    Get a profile token for a specific external ID.
    
    Args:
        external_id: External ID of the profile
        read_only: Whether the token should be read-only
        lifetime: Lifetime of the token in seconds
        
    Returns:
        Status message indicating success or failure
    """
    global PROFILE_TOKEN
    
    if not ACCOUNT_TOKEN:
        return "Account token is required to get a profile token. Run authenticate_account first."
    
    # Validate input parameters
    if error_msg := validate_not_empty(external_id, "external_id"):
        return error_msg
    
    if lifetime <= 0:
        return "Lifetime must be greater than zero"
    
    data = {
        "externalId": external_id,
        "readOnly": read_only,
        "lifetime": lifetime
    }
    
    response = await make_api_request(
        method="POST",
        endpoint="/api/v1/oauth/profile/token",
        data=data,
        auth_type="account"
    )
    
    if "error" in response and response["error"]:
        return f"Failed to get profile token: {response.get('message', 'Unknown error')}"
    
    if "profileToken" in response:
        PROFILE_TOKEN = response["profileToken"]
        return f"Successfully obtained profile token for external ID: {external_id}"
    else:
        return f"Failed to get profile token: {json.dumps(response)}"


@mcp.tool()
async def search_profiles(search_parameter: Optional[str] = None, current_page: int = 0, page_size: int = 10) -> str:
    """
    Search for profiles.
    
    Args:
        search_parameter: Text to search for in profiles
        current_page: Current page number (0-based)
        page_size: Number of results per page
        
    Returns:
        Formatted list of profiles or error message
    """
    if not ACCOUNT_TOKEN:
        return "Account token is required to search for profiles. Run authenticate_account first."
    
    # Validate input parameters
    if current_page < 0:
        return "current_page must be a non-negative integer"
    
    if page_size <= 0:
        return "page_size must be a positive integer"
    
    params = {
        "currentPage": current_page,
        "pageSize": page_size
    }
    
    if search_parameter:
        params["searchParameter"] = search_parameter
    
    response = await make_api_request(
        method="GET",
        endpoint="/api/v1/account/profile/search",
        params=params,
        auth_type="account"
    )
    
    if "error" in response and response["error"]:
        return f"Failed to search profiles: {response.get('message', 'Unknown error')}"
    
    # Format the response
    result = []
    
    if "items" in response and response["items"]:
        total_count = response.get("totalCount", len(response["items"]))
        current_page = response.get("currentPage", 0)
        total_pages = response.get("totalPages", 1)
        
        result.append(f"Found {total_count} profiles (Page {current_page + 1} of {total_pages})")
        
        for i, profile in enumerate(response["items"], 1):
            result.append(f"\n--- Profile {i} ---")
            result.append(f"External ID: {profile.get('externalId', 'N/A')}")
            
            if "createdAtUtc" in profile:
                created_date = profile["createdAtUtc"]
                result.append(f"Created: {created_date}")
            
            if "dataLastReceivedAtUtc" in profile and profile["dataLastReceivedAtUtc"]:
                last_data = profile["dataLastReceivedAtUtc"]
                result.append(f"Last Data: {last_data}")
            
            if "sdkId" in profile and profile["sdkId"]:
                result.append(f"SDK: {profile.get('sdkId')} {profile.get('sdkVersion', '')}")
            
            if "deviceType" in profile and profile["deviceType"]:
                result.append(f"Device: {profile.get('deviceType')}")
                
            if "isSampleProfile" in profile:
                result.append(f"Sample Profile: {'Yes' if profile['isSampleProfile'] else 'No'}")
    else:
        result.append("No profiles found")
    
    return "\n".join(result)


@mcp.tool()
async def get_comparisons_by_external_id(
    external_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    name: Optional[str] = None,
    category: Optional[str] = None,
    periodicity: str = "daily"
) -> str:
    """
    Get comparisons for a specific external ID.
    
    Args:
        external_id: External ID of the profile
        start_date: Start date in ISO format (e.g., "2023-05-01T00:00:00Z")
        end_date: End date in ISO format (e.g., "2023-05-31T23:59:59Z")
        name: Name to filter by
        category: Category to filter by
        periodicity: Periodicity to filter by (default: "daily")
        
    Returns:
        Formatted list of comparisons or error message
    """
    if not ACCOUNT_TOKEN:
        return "Account token is required to get comparisons by external ID. Run authenticate_account first."
    
    # Validate input parameters
    if error_msg := validate_not_empty(external_id, "external_id"):
        return error_msg
    
    if start_date and not validate_date_format(start_date):
        return "start_date must be in ISO format (e.g., '2023-05-01T00:00:00Z')"
    
    if end_date and not validate_date_format(end_date):
        return "end_date must be in ISO format (e.g., '2023-05-31T23:59:59Z')"
    
    params = {
        "periodicity": periodicity
    }
    
    if start_date:
        params["startDateTime"] = start_date
    
    if end_date:
        params["endDateTime"] = end_date
    
    if name:
        params["name"] = name
    
    if category:
        params["category"] = category
    
    response = await make_api_request(
        method="GET",
        endpoint=f"/api/v1/profile/insight/comparison/{external_id}",
        params=params,
        auth_type="account"
    )
    
    if "error" in response and response["error"]:
        return f"Failed to get comparisons for external ID {external_id}: {response.get('message', 'Unknown error')}"
    
    # Format the response using the helper function
    result = format_comparison_response(response)
    if result[0].startswith("No comparisons found"):
        result[0] = f"No comparisons found for external ID {external_id} with the specified criteria"
    else:
        # Update the first line to include the external ID
        result[0] = f"Found {len(response)} comparisons for external ID {external_id}:"
    
    return "\n".join(result)


@mcp.tool()
async def get_biomarkers_by_external_id(
    external_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    categories: Optional[str] = None,
    types: Optional[str] = None
) -> str:
    """
    Get biomarkers for a specific external ID.
    
    Args:
        external_id: External ID of the profile
        start_date: Start date in ISO format (e.g., "2023-05-01T00:00:00Z")
        end_date: End date in ISO format (e.g., "2023-05-31T23:59:59Z")
        categories: Comma-separated list of categories to filter by (e.g., "activity,sleep")
        types: Comma-separated list of types to filter by (e.g., "step_count,sleep_duration")
        
    Returns:
        Formatted list of biomarkers or error message
    """
    if not ACCOUNT_TOKEN:
        return "Account token is required to get biomarkers by external ID. Run authenticate_account first."
    
    # Validate input parameters
    if error_msg := validate_not_empty(external_id, "external_id"):
        return error_msg
    
    if start_date and not validate_date_format(start_date):
        return "start_date must be in ISO format (e.g., '2023-05-01T00:00:00Z')"
    
    if end_date and not validate_date_format(end_date):
        return "end_date must be in ISO format (e.g., '2023-05-31T23:59:59Z')"
    
    params = {}
    
    if start_date:
        params["startDateTime"] = start_date
    
    if end_date:
        params["endDateTime"] = end_date
    
    if categories:
        params["categories"] = categories.split(",")
    
    if types:
        params["types"] = types.split(",")
    
    response = await make_api_request(
        method="GET",
        endpoint=f"/api/v1/profile/biomarker/{external_id}",
        params=params,
        auth_type="account"
    )
    
    if "error" in response and response["error"]:
        return f"Failed to get biomarkers for external ID {external_id}: {response.get('message', 'Unknown error')}"
    
    # Format the response using the helper function
    result = format_biomarker_response(response)
    if result[0].startswith("No biomarkers found"):
        result[0] = f"No biomarkers found for external ID {external_id} with the specified criteria"
    else:
        # Update the first line to include the external ID
        result[0] = f"Found {len(response)} biomarkers for external ID {external_id}:"
    
    return "\n".join(result)


@mcp.tool()
async def get_archetypes_by_external_id(
    external_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    name: Optional[str] = None,
    periodicity: Optional[str] = None
) -> str:
    """
    Get archetypes for a specific external ID.
    
    Args:
        external_id: External ID of the profile
        start_date: Start date in ISO format (e.g., "2023-05-01T00:00:00Z")
        end_date: End date in ISO format (e.g., "2023-05-31T23:59:59Z")
        name: Name to filter by
        periodicity: Periodicity to filter by (e.g., "daily", "weekly")
        
    Returns:
        Formatted list of archetypes or error message
    """
    if not ACCOUNT_TOKEN:
        return "Account token is required to get archetypes by external ID. Run authenticate_account first."
    
    # Validate input parameters
    if error_msg := validate_not_empty(external_id, "external_id"):
        return error_msg
    
    if start_date and not validate_date_format(start_date):
        return "start_date must be in ISO format (e.g., '2023-05-01T00:00:00Z')"
    
    if end_date and not validate_date_format(end_date):
        return "end_date must be in ISO format (e.g., '2023-05-31T23:59:59Z')"
    
    params = {}
    
    if start_date:
        params["startDateTime"] = start_date
    
    if end_date:
        params["endDateTime"] = end_date
    
    if name:
        params["name"] = name
    
    if periodicity:
        params["periodicity"] = periodicity
    
    response = await make_api_request(
        method="GET",
        endpoint=f"/api/v1/profile/archetypes/{external_id}",
        params=params,
        auth_type="account"
    )
    
    if "error" in response and response["error"]:
        return f"Failed to get archetypes for external ID {external_id}: {response.get('message', 'Unknown error')}"
    
    # Format the response using the helper function
    result = format_archetype_response(response)
    if result[0].startswith("No archetypes found"):
        result[0] = f"No archetypes found for external ID {external_id} with the specified criteria"
    else:
        # Update the first line to include the external ID
        result[0] = f"Found {len(response)} archetypes for external ID {external_id}:"
    
    return "\n".join(result)


@mcp.tool()
async def get_integrations_by_external_id(external_id: str) -> str:
    """
    Get integrations for a specific external ID.
    
    Args:
        external_id: External ID of the profile
        
    Returns:
        Formatted list of integrations or error message
    """
    if not ACCOUNT_TOKEN:
        return "Account token is required to get integrations by external ID. Run authenticate_account first."
    
    # Validate input parameters
    if error_msg := validate_not_empty(external_id, "external_id"):
        return error_msg
    
    response = await make_api_request(
        method="GET",
        endpoint=f"/api/v1/profile/integration/{external_id}",
        auth_type="account"
    )
    
    if "error" in response and response["error"]:
        return f"Failed to get integrations for external ID {external_id}: {response.get('message', 'Unknown error')}"
    
    # Format the response using the helper function
    result = format_integration_response(response)
    if result[0].startswith("No integrations found"):
        result[0] = f"No integrations found for external ID {external_id}"
    else:
        # Update the first line to include the external ID
        result[0] = f"Found {len(response)} integrations for external ID {external_id}:"
    
    return "\n".join(result)


@mcp.tool()
async def get_trends_by_external_id(
    external_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    name: Optional[str] = None,
    category: Optional[str] = None,
    periodicity: str = "weekly"
) -> str:
    """
    Get trends for a specific external ID.
    
    Args:
        external_id: External ID of the profile
        start_date: Start date in ISO format (e.g., "2023-05-01T00:00:00Z")
        end_date: End date in ISO format (e.g., "2023-05-31T23:59:59Z")
        name: Name to filter by
        category: Category to filter by
        periodicity: Periodicity to filter by (default: "weekly")
        
    Returns:
        Formatted list of trends or error message
    """
    if not ACCOUNT_TOKEN:
        return "Account token is required to get trends by external ID. Run authenticate_account first."
    
    # Validate input parameters
    if error_msg := validate_not_empty(external_id, "external_id"):
        return error_msg
    
    if start_date and not validate_date_format(start_date):
        return "start_date must be in ISO format (e.g., '2023-05-01T00:00:00Z')"
    
    if end_date and not validate_date_format(end_date):
        return "end_date must be in ISO format (e.g., '2023-05-31T23:59:59Z')"
    
    params = {
        "periodicity": periodicity
    }
    
    if start_date:
        params["startDateTime"] = start_date
    
    if end_date:
        params["endDateTime"] = end_date
    
    if name:
        params["name"] = name
    
    if category:
        params["category"] = category
    
    response = await make_api_request(
        method="GET",
        endpoint=f"/api/v1/profile/insight/trend/{external_id}",
        params=params,
        auth_type="account"
    )
    
    if "error" in response and response["error"]:
        return f"Failed to get trends for external ID {external_id}: {response.get('message', 'Unknown error')}"
    
    # Format the response using the helper function
    result = format_trend_response(response)
    if result[0].startswith("No trends found"):
        result[0] = f"No trends found for external ID {external_id} with the specified criteria"
    else:
        # Update the first line to include the external ID
        result[0] = f"Found {len(response)} trends for external ID {external_id}:"
    
    return "\n".join(result)


@mcp.tool()
async def get_profile(external_id: str) -> str:
    """
    Get profile information by external ID.
    
    Args:
        external_id: External ID of the profile
        
    Returns:
        Formatted profile information or error message
    """
    if not ACCOUNT_TOKEN:
        return "Account token is required to get a profile. Run authenticate_account first."
    
    # Validate input parameters
    if error_msg := validate_not_empty(external_id, "external_id"):
        return error_msg
    
    response = await make_api_request(
        method="GET",
        endpoint=f"/api/v1/account/profile/{external_id}",
        auth_type="account"
    )
    
    if "error" in response and response["error"]:
        return f"Failed to get profile: {response.get('message', 'Unknown error')}"
    
    # Format the response
    result = [f"Profile Details for External ID: {external_id}"]
    
    if "profileId" in response:
        result.append(f"\nProfile ID: {response['profileId']}")
    
    if "accountId" in response:
        result.append(f"Account ID: {response['accountId']}")
    
    if "externalId" in response:
        result.append(f"External ID: {response['externalId']}")
    
    if "createdAtUtc" in response:
        result.append(f"Created: {response['createdAtUtc']}")
    
    if "dataLastReceivedAtUtc" in response and response["dataLastReceivedAtUtc"]:
        result.append(f"Last Data: {response['dataLastReceivedAtUtc']}")
    
    if "isSampleProfile" in response:
        result.append(f"Sample Profile: {'Yes' if response['isSampleProfile'] else 'No'}")
    
    if "dataSources" in response and response["dataSources"]:
        result.append("\nData Sources:")
        for i, source in enumerate(response["dataSources"], 1):
            result.append(f"  {i}. {source.get('sourceName', 'N/A')} (Device: {source.get('deviceType', 'N/A')})")
    
    return "\n".join(result)


@mcp.tool()
async def get_profile_info() -> str:
    """
    Get profile information for the authenticated profile.
    
    Returns:
        Formatted profile information or error message
    """
    if not PROFILE_TOKEN:
        return "Profile token is required to get profile information. Run get_profile_token first."
    
    response = await make_api_request(
        method="GET",
        endpoint="/api/v1/profile",
        auth_type="profile"
    )
    
    if "error" in response and response["error"]:
        return f"Failed to get profile information: {response.get('message', 'Unknown error')}"
    
    # Format the response
    return f"Profile Information:\n{json.dumps(response, indent=2)}"


@mcp.tool()
async def get_demographic_info() -> str:
    """
    Get demographic information for the authenticated profile.
    
    Returns:
        Formatted demographic information or error message
    """
    if not PROFILE_TOKEN:
        return "Profile token is required to get demographic information. Run get_profile_token first."
    
    response = await make_api_request(
        method="GET",
        endpoint="/api/v1/profile/demographic",
        auth_type="profile"
    )
    
    if "error" in response and response["error"]:
        return f"Failed to get demographic information: {response.get('message', 'Unknown error')}"
    
    # Format the response
    result = ["Demographic Information:"]
    
    if "age" in response and response["age"]:
        result.append(f"Age: {response['age']}")
    
    if "gender" in response and response["gender"]:
        result.append(f"Gender: {response['gender']}")
    
    if "birthDate" in response and response["birthDate"]:
        result.append(f"Birth Date: {response['birthDate']}")
    
    if len(result) == 1:
        result.append("No demographic information available")
    
    return "\n".join(result)


@mcp.tool()
async def get_device_info() -> str:
    """
    Get device information for the authenticated profile.
    
    Returns:
        Formatted device information or error message
    """
    if not PROFILE_TOKEN:
        return "Profile token is required to get device information. Run get_profile_token first."
    
    response = await make_api_request(
        method="GET",
        endpoint="/api/v1/profile/deviceInformation",
        auth_type="profile"
    )
    
    if "error" in response and response["error"]:
        return f"Failed to get device information: {response.get('message', 'Unknown error')}"
    
    # Format the response
    result = ["Device Information:"]
    
    if "deviceType" in response and response["deviceType"]:
        result.append(f"Device Type: {response['deviceType']}")
    
    if "deviceModel" in response and response["deviceModel"]:
        result.append(f"Device Model: {response['deviceModel']}")
    
    if "system" in response and response["system"]:
        result.append(f"System: {response['system']}")
    
    if "systemVersion" in response and response["systemVersion"]:
        result.append(f"System Version: {response['systemVersion']}")
    
    if "sdkId" in response and response["sdkId"]:
        result.append(f"SDK ID: {response['sdkId']}")
    
    if "sdkVersion" in response and response["sdkVersion"]:
        result.append(f"SDK Version: {response['sdkVersion']}")
    
    if "timeZone" in response and response["timeZone"]:
        result.append(f"Time Zone: {response['timeZone']}")
    
    if len(result) == 1:
        result.append("No device information available")
    
    return "\n".join(result)


@mcp.tool()
async def get_integrations() -> str:
    """
    Get integrations for the authenticated profile.
    
    Returns:
        Formatted list of integrations or error message
    """
    if not PROFILE_TOKEN:
        return "Profile token is required to get integrations. Run get_profile_token first."
    
    response = await make_api_request(
        method="GET",
        endpoint="/api/v1/profile/integration",
        auth_type="profile"
    )
    
    if "error" in response and response["error"]:
        return f"Failed to get integrations: {response.get('message', 'Unknown error')}"
    
    # Format the response using the helper function
    return "\n".join(format_integration_response(response))


@mcp.tool()
async def get_biomarkers(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    categories: Optional[str] = None,
    types: Optional[str] = None
) -> str:
    """
    Get biomarkers for the authenticated profile.
    
    Args:
        start_date: Start date in ISO format (e.g., "2023-05-01T00:00:00Z")
        end_date: End date in ISO format (e.g., "2023-05-31T23:59:59Z")
        categories: Comma-separated list of categories to filter by (e.g., "activity,sleep")
        types: Comma-separated list of types to filter by (e.g., "step_count,sleep_duration")
        
    Returns:
        Formatted list of biomarkers or error message
    """
    if not PROFILE_TOKEN:
        return "Profile token is required to get biomarkers. Run get_profile_token first."
    
    # Validate input parameters
    if start_date and not validate_date_format(start_date):
        return "start_date must be in ISO format (e.g., '2023-05-01T00:00:00Z')"
    
    if end_date and not validate_date_format(end_date):
        return "end_date must be in ISO format (e.g., '2023-05-31T23:59:59Z')"
    
    params = {}
    
    if start_date:
        params["startDateTime"] = start_date
    
    if end_date:
        params["endDateTime"] = end_date
    
    if categories:
        params["categories"] = categories.split(",")
    
    if types:
        params["types"] = types.split(",")
    
    response = await make_api_request(
        method="GET",
        endpoint="/api/v1/profile/biomarker",
        params=params,
        auth_type="profile"
    )
    
    if "error" in response and response["error"]:
        return f"Failed to get biomarkers: {response.get('message', 'Unknown error')}"
    
    # Format the response using the helper function
    return "\n".join(format_biomarker_response(response))


@mcp.tool()
async def get_archetypes(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    name: Optional[str] = None,
    periodicity: Optional[str] = None
) -> str:
    """
    Get archetypes for the authenticated profile.
    
    Args:
        start_date: Start date in ISO format (e.g., "2023-05-01T00:00:00Z")
        end_date: End date in ISO format (e.g., "2023-05-31T23:59:59Z")
        name: Name to filter by
        periodicity: Periodicity to filter by (e.g., "daily", "weekly")
        
    Returns:
        Formatted list of archetypes or error message
    """
    if not PROFILE_TOKEN:
        return "Profile token is required to get archetypes. Run get_profile_token first."
    
    # Validate input parameters
    if start_date and not validate_date_format(start_date):
        return "start_date must be in ISO format (e.g., '2023-05-01T00:00:00Z')"
    
    if end_date and not validate_date_format(end_date):
        return "end_date must be in ISO format (e.g., '2023-05-31T23:59:59Z')"
    
    params = {}
    
    if start_date:
        params["startDateTime"] = start_date
    
    if end_date:
        params["endDateTime"] = end_date
    
    if name:
        params["name"] = name
    
    if periodicity:
        params["periodicity"] = periodicity
    
    response = await make_api_request(
        method="GET",
        endpoint="/api/v1/profile/archetypes",
        params=params,
        auth_type="profile"
    )
    
    if "error" in response and response["error"]:
        return f"Failed to get archetypes: {response.get('message', 'Unknown error')}"
    
    # Format the response using the helper function
    return "\n".join(format_archetype_response(response))


@mcp.tool()
async def get_trends(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    name: Optional[str] = None,
    category: Optional[str] = None,
    periodicity: str = "weekly"
) -> str:
    """
    Get trends for the authenticated profile.
    
    Args:
        start_date: Start date in ISO format (e.g., "2023-05-01T00:00:00Z")
        end_date: End date in ISO format (e.g., "2023-05-31T23:59:59Z")
        name: Name to filter by
        category: Category to filter by
        periodicity: Periodicity to filter by (default: "weekly")
        
    Returns:
        Formatted list of trends or error message
    """
    if not PROFILE_TOKEN:
        return "Profile token is required to get trends. Run get_profile_token first."
    
    # Validate input parameters
    if start_date and not validate_date_format(start_date):
        return "start_date must be in ISO format (e.g., '2023-05-01T00:00:00Z')"
    
    if end_date and not validate_date_format(end_date):
        return "end_date must be in ISO format (e.g., '2023-05-31T23:59:59Z')"
    
    params = {
        "periodicity": periodicity
    }
    
    if start_date:
        params["startDateTime"] = start_date
    
    if end_date:
        params["endDateTime"] = end_date
    
    if name:
        params["name"] = name
    
    if category:
        params["category"] = category
    
    response = await make_api_request(
        method="GET",
        endpoint="/api/v1/profile/insight/trend",
        params=params,
        auth_type="profile"
    )
    
    if "error" in response and response["error"]:
        return f"Failed to get trends: {response.get('message', 'Unknown error')}"
    
    # Format the response using the helper function
    return "\n".join(format_trend_response(response))


@mcp.tool()
async def get_comparisons(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    name: Optional[str] = None,
    category: Optional[str] = None,
    periodicity: str = "daily"
) -> str:
    """
    Get comparisons for the authenticated profile.
    
    Args:
        start_date: Start date in ISO format (e.g., "2023-05-01T00:00:00Z")
        end_date: End date in ISO format (e.g., "2023-05-31T23:59:59Z")
        name: Name to filter by
        category: Category to filter by
        periodicity: Periodicity to filter by (default: "daily")
        
    Returns:
        Formatted list of comparisons or error message
    """
    if not PROFILE_TOKEN:
        return "Profile token is required to get comparisons. Run get_profile_token first."
    
    # Validate input parameters
    if start_date and not validate_date_format(start_date):
        return "start_date must be in ISO format (e.g., '2023-05-01T00:00:00Z')"
    
    if end_date and not validate_date_format(end_date):
        return "end_date must be in ISO format (e.g., '2023-05-31T23:59:59Z')"
    
    params = {
        "periodicity": periodicity
    }
    
    if start_date:
        params["startDateTime"] = start_date
    
    if end_date:
        params["endDateTime"] = end_date
    
    if name:
        params["name"] = name
    
    if category:
        params["category"] = category
    
    response = await make_api_request(
        method="GET",
        endpoint="/api/v1/profile/insight/comparison",
        params=params,
        auth_type="profile"
    )
    
    if "error" in response and response["error"]:
        return f"Failed to get comparisons: {response.get('message', 'Unknown error')}"
    
    # Format the response using the helper function
    return "\n".join(format_comparison_response(response))


def main():
    """Run the MCP server."""
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()