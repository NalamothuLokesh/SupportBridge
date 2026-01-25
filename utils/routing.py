import pandas as pd
from datetime import datetime, timedelta
import random


class TicketRouter:
    """
    Routes tickets to appropriate support teams based on category and priority.
    Provides assignment logic and SLA management.
    """
    
    TEAM_MAPPING = {
        'Account': 'Account Support Team',
        'Billing': 'Billing & Finance Team',
        'Technical': 'Technical Support Team',
        'Feature Request': 'Product Team'
    }
    
    SLA_TIMES = {
        'Critical': 1,      # 1 hour
        'High': 4,          # 4 hours
        'Medium': 8,        # 8 hours
        'Low': 24           # 24 hours
    }
    
    TEAM_MEMBERS = {
        'Account Support Team': ['Alice Johnson', 'Bob Smith', 'Carol Davis'],
        'Billing & Finance Team': ['David Wilson', 'Emma Brown', 'Frank Miller'],
        'Technical Support Team': ['Grace Lee', 'Henry Taylor', 'Iris Martinez'],
        'Product Team': ['Jack Anderson', 'Karen Thomas', 'Leo Jackson']
    }
    
    @classmethod
    def get_team(cls, category):
        """
        Get the appropriate team for a category.
        
        Args:
            category: Ticket category
            
        Returns:
            Team name
        """
        return cls.TEAM_MAPPING.get(category, 'General Support Team')
    
    @classmethod
    def get_assignee(cls, category):
        """
        Assign ticket to a team member.
        
        Args:
            category: Ticket category
            
        Returns:
            Team member name
        """
        team = cls.get_team(category)
        members = cls.TEAM_MEMBERS.get(team, ['Unassigned'])
        return random.choice(members)
    
    @classmethod
    def calculate_sla(cls, priority):
        """
        Calculate SLA deadline for a ticket.
        
        Args:
            priority: Ticket priority level
            
        Returns:
            SLA deadline datetime
        """
        hours = cls.SLA_TIMES.get(priority, 24)
        return datetime.now() + timedelta(hours=hours)
    
    @classmethod
    def route_ticket(cls, ticket_data):
        """
        Complete routing decision for a ticket.
        
        Args:
            ticket_data: Dictionary with 'category' and 'priority'
            
        Returns:
            Dictionary with routing information
        """
        category = ticket_data.get('category', 'Unknown')
        priority = ticket_data.get('priority', 'Low')
        confidence = ticket_data.get('confidence', 0.0)
        
        team = cls.get_team(category)
        assignee = cls.get_assignee(category)
        sla_deadline = cls.calculate_sla(priority)
        
        routing_info = {
            'team': team,
            'assignee': assignee,
            'sla_deadline': sla_deadline.strftime('%Y-%m-%d %H:%M:%S'),
            'sla_hours': cls.SLA_TIMES.get(priority, 24),
            'confidence_score': round(confidence, 4),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return routing_info
    
    @classmethod
    def get_routing_metrics(cls, routing_history):
        """
        Calculate metrics from routing history.
        
        Args:
            routing_history: List of routing decisions
            
        Returns:
            Dictionary with metrics
        """
        if not routing_history:
            return {}
        
        df = pd.DataFrame(routing_history)
        
        metrics = {
            'total_tickets': len(df),
            'avg_confidence': df['confidence_score'].mean(),
            'tickets_by_priority': df['priority'].value_counts().to_dict() if 'priority' in df else {},
            'tickets_by_category': df['category'].value_counts().to_dict() if 'category' in df else {},
            'tickets_by_team': df['team'].value_counts().to_dict() if 'team' in df else {}
        }
        
        return metrics
    
    @classmethod
    def suggest_team_allocation(cls, routing_history):
        """
        Suggest optimal team size based on ticket distribution.
        
        Args:
            routing_history: List of routing decisions
            
        Returns:
            Dictionary with team allocation recommendations
        """
        metrics = cls.get_routing_metrics(routing_history)
        
        if not metrics.get('tickets_by_team'):
            return {}
        
        total = metrics['total_tickets']
        allocation = {}
        
        for team, count in metrics['tickets_by_team'].items():
            percentage = (count / total) * 100
            # Recommend team size (1-5 members) based on ticket volume
            recommended_size = max(1, min(5, round(count / 10)))
            allocation[team] = {
                'tickets': count,
                'percentage': round(percentage, 2),
                'recommended_size': recommended_size
            }
        
        return allocation


# ============================================================================
# Module-level wrapper functions for backward compatibility
# ============================================================================

def assign_agent(category):
    """Wrapper function to assign agent based on category."""
    return TicketRouter.get_assignee(category)

def calculate_sla(priority):
    """Wrapper function to calculate SLA hours for priority level."""
    return TicketRouter.SLA_TIMES.get(priority, 24)

def route_ticket(ticket_data):
    """Wrapper function for complete ticket routing."""
    return TicketRouter.route_ticket(ticket_data)
