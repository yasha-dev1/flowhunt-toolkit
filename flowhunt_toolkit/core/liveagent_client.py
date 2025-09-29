"""LiveAgent API client for fetching and processing tickets."""

import requests
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class LiveAgentClient:
    """Client for interacting with LiveAgent API."""
    
    def __init__(self, base_url: str, api_key: str):
        """Initialize LiveAgent client.
        
        Args:
            base_url: Base URL of LiveAgent instance (e.g., https://support.qualityunit.com)
            api_key: LiveAgent API key (read-only for tickets)
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'apikey': api_key,
            'Accept': 'application/json'
        }
    
    def get_tickets(self, department_id: Optional[str] = None, limit: int = 100, 
                   offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """Fetch tickets from LiveAgent with pagination.
        
        Args:
            department_id: Optional department ID to filter tickets
            limit: Maximum number of tickets per page
            offset: Pagination offset
            
        Returns:
            Tuple of (tickets list, total count)
        """
        url = f"{self.base_url}/api/v3/tickets"
        
        params = {
            '_perPage': 100,  # LiveAgent max is typically 100 per page
            '_page': (offset // 100) + 1,
            '_sortField': 'date_created',
            '_sortDir': 'DESC',
            'status': 'R,L',  # Only Resolved/Closed tickets
            'channel': 'M'  # Email tickets only
        }
        
        # Add department filtering using _filters parameter
        if department_id:
            params['_filters'] = json.dumps({"departmentid": department_id})
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # LiveAgent returns a list directly, not an object with 'tickets' key
            if isinstance(data, list):
                # If it's a list, the tickets are the data itself
                tickets = data
                # Try to get total from response headers or use list length
                total = len(tickets)
            else:
                # If it's an object, extract tickets and total
                tickets = data.get('tickets', data.get('response', []))
                total = data.get('_total', data.get('total', len(tickets)))
            
            # Note: Department filtering should be done after fetching
            # to avoid blocking during the fetch phase
            return tickets, total
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch tickets: {str(e)}")
    
    def get_ticket_conversation(self, ticket_id: str) -> Dict[str, Any]:
        """Fetch full conversation for a specific ticket.
        
        Args:
            ticket_id: Ticket ID to fetch conversation for
            
        Returns:
            Full ticket data including messages
        """
        url = f"{self.base_url}/api/v3/tickets/{ticket_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch ticket {ticket_id}: {str(e)}")
    
    def get_ticket_messages(self, ticket_id: str) -> List[Dict[str, Any]]:
        """Fetch all messages for a ticket.
        
        Args:
            ticket_id: Ticket ID to fetch messages for
            
        Returns:
            List of message group objects with their messages
        """
        url = f"{self.base_url}/api/v3/tickets/{ticket_id}/messages"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            # The API returns a list of message groups
            if isinstance(data, list):
                return data
            else:
                return []
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch messages for ticket {ticket_id}: {str(e)}")
    
    def format_ticket_as_text(self, ticket: Dict[str, Any], message_groups: List[Dict[str, Any]]) -> str:
        """Format ticket and its messages as structured text.
        
        Args:
            ticket: Ticket object from API
            message_groups: List of message group objects
            
        Returns:
            Formatted text representation of the ticket
        """
        lines = []
        
        # Ticket metadata
        lines.append(f"=== TICKET #{ticket.get('code', 'N/A')} ===")
        lines.append(f"Subject: {ticket.get('subject', 'No subject')}")
        lines.append(f"Status: {ticket.get('status', 'Unknown')}")
        lines.append(f"Department: {ticket.get('department_name', 'N/A')}")
        lines.append(f"Created: {ticket.get('date_created', 'N/A')}")
        lines.append(f"Customer: {ticket.get('customer_email', 'N/A')} ({ticket.get('customer_name', 'N/A')})")
        
        # Tags if present
        tags = ticket.get('tags', [])
        if tags:
            lines.append(f"Tags: {', '.join(tags)}")
        
        lines.append("\n=== CONVERSATION ===")
        
        # Process message groups
        for group in message_groups:
            group_date = group.get('datecreated', 'N/A')
            group_user = group.get('user_full_name', '')
            group_userid = group.get('userid', '')
            group_type = group.get('type', '')
            
            # Process messages within the group
            group_messages = group.get('messages', [])
            
            if group_messages:
                # Find the main message (type 'M') and headers, skip notes (type 'N')
                main_message = None
                subject = None
                from_header = None
                
                for msg in group_messages:
                    msg_type = msg.get('type', '')
                    msg_text = msg.get('message', '').strip()
                    
                    if msg_type == 'M':  # Main message content
                        main_message = msg_text
                    elif msg_type == 'H':  # Header
                        if msg_text.startswith('Subject:'):
                            subject = msg_text.replace('Subject:', '').strip()
                        elif msg_text.startswith('From:'):
                            from_header = msg_text.replace('From:', '').strip()
                    # Skip type 'N' (notes) and type 'T' (internal tracking)
                
                # Skip system messages that only contain notes (no actual content)
                if (group_userid == 'system00' or group_type == 'I') and not main_message:
                    continue
                
                # Format the conversation entry
                lines.append(f"\n--- {group_date} ---")
                
                # Determine sender
                if group_userid == 'system00' or group_type == 'I':
                    sender = "System/Agent"
                elif group_user:
                    sender = f"Agent ({group_user})"
                elif from_header:
                    sender = f"Customer ({from_header})"
                else:
                    sender = f"User ({group_userid})"
                
                lines.append(f"From: {sender}")
                
                if subject and subject != ticket.get('subject'):
                    lines.append(f"Subject: {subject}")
                
                # Add the main message
                if main_message:
                    # Clean HTML if present
                    import re
                    clean_msg = re.sub(r'<[^>]+>', '', main_message)
                    clean_msg = clean_msg.replace('&nbsp;', ' ')
                    clean_msg = clean_msg.replace('&lt;', '<')
                    clean_msg = clean_msg.replace('&gt;', '>')
                    clean_msg = clean_msg.replace('&amp;', '&')
                    clean_msg = clean_msg.strip()
                    
                    if clean_msg:
                        lines.append(f"Message:\n{clean_msg}")
                    else:
                        lines.append("Message: [Empty message]")
                else:
                    lines.append("Message: [No content]")
        
        return '\n'.join(lines)
    
    def paginate_all_tickets(self, department_id: Optional[str] = None,
                            max_tickets: Optional[int] = None,
                            skip_ids: Optional[set] = None) -> List[Dict[str, Any]]:
        """Fetch all tickets with automatic pagination.

        Args:
            department_id: Optional department ID to filter tickets
            max_tickets: Maximum total tickets to fetch (None for all)
            skip_ids: Set of ticket IDs to skip (for resume functionality)

        Returns:
            List of all ticket objects
        """
        all_tickets = []
        offset = 0
        page_size = 100
        skip_ids = skip_ids or set()
        total_fetched = 0

        while True:
            tickets, total = self.get_tickets(
                department_id=department_id,  # Pass department_id for API filtering
                limit=page_size,
                offset=offset
            )

            if not tickets:
                break

            # Filter out tickets we should skip
            new_tickets = []
            for ticket in tickets:
                ticket_id = str(ticket.get('id', ''))
                if ticket_id not in skip_ids:
                    new_tickets.append(ticket)

            all_tickets.extend(new_tickets)
            total_fetched += len(tickets)

            # Check if we've reached the desired number of NEW tickets
            if max_tickets and len(all_tickets) >= max_tickets:
                all_tickets = all_tickets[:max_tickets]
                break

            # If we got fewer tickets than page_size, we've reached the end of available tickets
            if len(tickets) < page_size:
                break

            offset += page_size

            # Rate limiting - be nice to the API
            time.sleep(0.5)

        return all_tickets