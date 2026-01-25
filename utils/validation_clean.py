"""
Text Validation Module for Ticket Input
Validates ticket subject and description for appropriate content
"""

import re

# List of inappropriate/profane words
INAPPROPRIATE_WORDS = [
    'damn', 'hell', 'crap', 'stupid', 'idiot', 'jerk', 'ass',
    'bastard', 'bitch', 'crap', 'fuck', 'shit', 'piss', 'ass',
    'retard', 'insult', 'offensive'
]

# Customer support keywords to validate context
SUPPORT_KEYWORDS = {
    'access': ['login', 'access', 'sign in', 'sign-in', 'password', 'account', 'credentials'],
    'technical': ['error', 'bug', 'issue', 'crash', 'problem', 'not working', 'broken', 'failed', 'failure', 'malfunction'],
    'feature': ['feature', 'request', 'function', 'functionality', 'capability', 'option', 'button'],
    'payment': ['payment', 'billing', 'charge', 'invoice', 'refund', 'subscription', 'credit card', 'transaction'],
    'account': ['account', 'profile', 'settings', 'password', 'email', 'username', 'user'],
    'data': ['data', 'information', 'file', 'upload', 'download', 'export', 'import', 'sync'],
    'support': ['help', 'support', 'assist', 'question', 'assistance', 'urgent', 'critical'],
    'performance': ['slow', 'lag', 'delay', 'freeze', 'hang', 'performance', 'speed'],
    'integration': ['integration', 'connect', 'api', 'sync', 'plugin', 'extension', 'module'],
    'notification': ['notification', 'alert', 'message', 'email', 'notification', 'reminder']
}

# Irrelevant keywords that suggest non-support content
IRRELEVANT_KEYWORDS = [
    'weather', 'sports', 'news', 'movie', 'game', 'book', 'recipe', 'cooking',
    'music', 'celebrity', 'political', 'politics', 'religion', 'joke', 'meme',
    'random', 'test', 'hello world', 'foo', 'bar', 'lorem ipsum'
]


def contains_numbers(text):
    """
    Check if text contains numeric digits.
    
    Args:
        text: Input text to check
        
    Returns:
        tuple: (has_numbers: bool, first_number: str or None)
    """
    numbers = re.findall(r'\d', text)
    if numbers:
        return True, numbers[0]
    return False, None


def contains_special_characters(text):
    """
    Check if text contains excessive/dangerous special characters.
    Allows common characters used in real ticket descriptions.
    
    Args:
        text: Input text to check
        
    Returns:
        tuple: (has_dangerous_chars: bool, characters: list)
    """
    # Allow letters, numbers, spaces, hyphens, underscores, periods, commas, 
    # apostrophes, question marks, exclamation marks, parentheses, forward slashes, colons, semicolons
    # Do NOT allow: @, #, $, %, ^, &, *, ~, `, <, >, |, =, {}, [], etc.
    allowed_pattern = r"^[a-zA-Z0-9\s\-_.,:;'\"()/!?]+$"
    
    if not re.match(allowed_pattern, text):
        # Find characters that aren't allowed
        dangerous_chars = re.findall(r'[^a-zA-Z0-9\s\-_.,:;\'\"()/!?]', text)
        return True, list(set(dangerous_chars))
    
    return False, []


def is_gibberish_text(text):
    """
    Detect if text is gibberish or random characters (e.g., 'jhfdbds').
    Real text should have a reasonable number of vowels and not be just random characters.
    
    Args:
        text: Input text to check
        
    Returns:
        tuple: (is_gibberish: bool, reason: str or None)
    """
    # Count vowels and consonants
    vowels = 'aeiouAEIOU'
    vowel_count = sum(1 for char in text if char in vowels)
    
    # Remove spaces and count only letters
    letters_only = ''.join(char for char in text if char.isalpha())
    
    if len(letters_only) == 0:
        return False, None  # No letters, will be caught by other validations
    
    # Check if text has very few or no vowels (e.g., "jhfdbds")
    if vowel_count == 0:
        return True, "Text must contain meaningful words (detected gibberish with no vowels)"
    
    # Check vowel ratio - real text typically has 30-40% vowels
    vowel_ratio = vowel_count / len(letters_only) if len(letters_only) > 0 else 0
    if vowel_ratio < 0.15:  # Less than 15% vowels is suspicious
        return True, "Text appears to be random characters or gibberish (too few vowels)"
    
    # Check for excessive repetition of same character (e.g., "aaaaaaa", "xxxx")
    for i in range(len(text) - 2):
        if text[i] == text[i+1] == text[i+2] and text[i].isalpha():
            # Found 3+ repeated letters, check if it's more than 30% of text
            char = text[i]
            char_count = text.lower().count(char.lower())
            if char_count / len(letters_only) > 0.3:
                return True, f"Text has excessive repetition ('{char}' appears too many times)"
    
    return False, None


def contains_inappropriate_words(text):
    """
    Check if text contains inappropriate/profane words.
    
    Args:
        text: Input text to check
        
    Returns:
        tuple: (has_inappropriate: bool, found_words: list)
    """
    text_lower = text.lower()
    found_words = []
    
    for word in INAPPROPRIATE_WORDS:
        if re.search(r'\b' + word + r'\b', text_lower):
            found_words.append(word)
    
    return len(found_words) > 0, found_words


def contains_support_keywords(text):
    """
    Check if text contains customer support-relevant keywords.
    
    Args:
        text: Input text to check
        
    Returns:
        tuple: (has_keywords: bool, found_categories: list, keyword_list: list)
    """
    text_lower = text.lower()
    found_categories = []
    found_keywords = []
    
    for category, keywords in SUPPORT_KEYWORDS.items():
        for keyword in keywords:
            if re.search(r'\b' + keyword + r'\b', text_lower):
                if category not in found_categories:
                    found_categories.append(category)
                found_keywords.append(keyword)
    
    return len(found_keywords) > 0, found_categories, list(set(found_keywords))


def contains_irrelevant_keywords(text):
    """
    Check if text contains non-support related keywords.
    
    Args:
        text: Input text to check
        
    Returns:
        tuple: (has_irrelevant: bool, found_keywords: list)
    """
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in IRRELEVANT_KEYWORDS:
        if re.search(r'\b' + keyword + r'\b', text_lower):
            found_keywords.append(keyword)
    
    return len(found_keywords) > 0, found_keywords


def validate_ticket_id(ticket_id):
    """
    Validate ticket ID format.
    Accepts format like: TKT-001, TICKET-123, ID-456, etc.
    
    Args:
        ticket_id: Ticket ID to validate
        
    Returns:
        dict: {
            'is_valid': bool,
            'error_message': str or None,
            'error_type': str or None
        }
    """
    # Check if empty
    if not ticket_id or not ticket_id.strip():
        return {
            'is_valid': False,
            'error_message': "[ERROR] Ticket ID cannot be empty.",
            'error_type': 'empty'
        }
    
    ticket_id = ticket_id.strip()
    
    # Check if too long
    if len(ticket_id) > 50:
        return {
            'is_valid': False,
            'error_message': "[ERROR] Ticket ID cannot exceed 50 characters.",
            'error_type': 'too_long'
        }
    
    # Validate format: must contain alphanumeric and hyphens, underscores only
    # Examples: TKT-001, TICKET-123, ID_456
    if not re.match(r'^[a-zA-Z0-9\-_]+$', ticket_id):
        return {
            'is_valid': False,
            'error_message': "[ERROR] Ticket ID can only contain letters, numbers, hyphens, and underscores (e.g., TKT-001).",
            'error_type': 'invalid_format'
        }
    
    # Check that it's not just numbers (should have some prefix like TKT-)
    if re.match(r'^\d+$', ticket_id):
        return {
            'is_valid': False,
            'error_message': "[ERROR] Ticket ID should have a prefix (e.g., TKT-001 instead of just 001).",
            'error_type': 'invalid_format'
        }
    
    return {
        'is_valid': True,
        'error_message': None,
        'error_type': None
    }


def is_valid_email_format(text):
    """
    Check if text looks like an email address (for subject/description).
    Usually subject/description shouldn't be just an email.
    
    Args:
        text: Input text to check
        
    Returns:
        bool: True if text is primarily an email
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, text.strip()))


def is_text_too_short(text):
    """
    Check if text is too short to be meaningful.
    
    Args:
        text: Input text to check
        
    Returns:
        bool: True if text is less than 3 characters
    """
    return len(text.strip()) < 3


def is_text_too_long(text, max_length=500):
    """
    Check if text exceeds maximum length.
    
    Args:
        text: Input text to check
        max_length: Maximum allowed length (default 500)
        
    Returns:
        bool: True if text exceeds max_length
    """
    return len(text) > max_length


def validate_subject(subject):
    """
    Validate ticket subject.
    
    Args:
        subject: Ticket subject text
        
    Returns:
        dict: {
            'is_valid': bool,
            'error_message': str or None,
            'error_type': str or None
        }
    """
    # Check if empty
    if not subject or not subject.strip():
        return {
            'is_valid': False,
            'error_message': "[ERROR] Subject cannot be empty.",
            'error_type': 'empty'
        }
    
    # Check if too short
    if is_text_too_short(subject):
        return {
            'is_valid': False,
            'error_message': "[ERROR] Subject must be at least 3 characters long.",
            'error_type': 'too_short'
        }
    
    # Check if too long
    if is_text_too_long(subject, max_length=150):
        return {
            'is_valid': False,
            'error_message': "[ERROR] Subject cannot exceed 150 characters.",
            'error_type': 'too_long'
        }
    
    # Check for special characters (now allows numbers and common punctuation)
    has_special, chars = contains_special_characters(subject)
    if has_special:
        invalid_chars_str = ', '.join([f"'{c}'" for c in chars[:3]])
        return {
            'is_valid': False,
            'error_message': f"[ERROR] Invalid characters not allowed: {invalid_chars_str}. Please use alphanumeric, spaces, and common punctuation only.",
            'error_type': 'invalid_characters'
        }
    
    # Check for gibberish text
    is_gibberish, reason = is_gibberish_text(subject)
    if is_gibberish:
        return {
            'is_valid': False,
            'error_message': f"[ERROR] {reason}. Please provide a meaningful subject.",
            'error_type': 'gibberish'
        }
    
    # Check for inappropriate words
    has_inappropriate, found_words = contains_inappropriate_words(subject)
    if has_inappropriate:
        return {
            'is_valid': False,
            'error_message': f"[ERROR] Please avoid using offensive language in your subject.",
            'error_type': 'inappropriate_language'
        }
    
    # Check for irrelevant keywords (non-support content)
    has_irrelevant, irrelevant_words = contains_irrelevant_keywords(subject)
    if has_irrelevant:
        return {
            'is_valid': False,
            'error_message': f"[ERROR] Subject should be about a customer support issue, not about {irrelevant_words[0]}. Please describe your support request.",
            'error_type': 'irrelevant_content'
        }
    
    # Check for support keywords - must contain at least one
    has_keywords, categories, keywords = contains_support_keywords(subject)
    if not has_keywords:
        return {
            'is_valid': False,
            'error_message': "[ERROR] Subject must contain keywords related to customer support (e.g., 'login issue', 'payment problem', 'feature request', 'error', 'bug', etc.).",
            'error_type': 'missing_keywords'
        }
    
    return {
        'is_valid': True,
        'error_message': None,
        'error_type': None
    }


def validate_description(description):
    """
    Validate ticket description.
    
    Args:
        description: Ticket description text
        
    Returns:
        dict: {
            'is_valid': bool,
            'error_message': str or None,
            'error_type': str or None
        }
    """
    # Check if empty
    if not description or not description.strip():
        return {
            'is_valid': False,
            'error_message': "[ERROR] Description cannot be empty.",
            'error_type': 'empty'
        }
    
    # Check if too short
    if is_text_too_short(description):
        return {
            'is_valid': False,
            'error_message': "[ERROR] Description must be at least 3 characters long.",
            'error_type': 'too_short'
        }
    
    # Check if too long
    if is_text_too_long(description, max_length=2000):
        return {
            'is_valid': False,
            'error_message': "[ERROR] Description cannot exceed 2000 characters.",
            'error_type': 'too_long'
        }
    
    # Check for special characters (now allows numbers and common punctuation)
    has_special, chars = contains_special_characters(description)
    if has_special:
        invalid_chars_str = ', '.join([f"'{c}'" for c in chars[:3]])
        return {
            'is_valid': False,
            'error_message': f"[ERROR] Invalid characters not allowed: {invalid_chars_str}. Please use alphanumeric, spaces, and common punctuation only.",
            'error_type': 'invalid_characters'
        }
    
    # Check for gibberish text
    is_gibberish, reason = is_gibberish_text(description)
    if is_gibberish:
        return {
            'is_valid': False,
            'error_message': f"[ERROR] {reason}. Please provide a meaningful description.",
            'error_type': 'gibberish'
        }
    
    # Check for inappropriate words
    has_inappropriate, found_words = contains_inappropriate_words(description)
    if has_inappropriate:
        return {
            'is_valid': False,
            'error_message': f"[ERROR] Please avoid using offensive language in your description.",
            'error_type': 'inappropriate_language'
        }
    
    # Check for irrelevant keywords (non-support content)
    has_irrelevant, irrelevant_words = contains_irrelevant_keywords(description)
    if has_irrelevant:
        return {
            'is_valid': False,
            'error_message': f"[ERROR] Description should be about a customer support issue, not about {irrelevant_words[0]}. Please describe your support request.",
            'error_type': 'irrelevant_content'
        }
    
    # Check for support keywords - must contain at least one
    has_keywords, categories, keywords = contains_support_keywords(description)
    if not has_keywords:
        return {
            'is_valid': False,
            'error_message': "[ERROR] Description must contain keywords related to customer support (e.g., 'issue', 'error', 'problem', 'not working', 'help needed', 'account', 'payment', etc.).",
            'error_type': 'missing_keywords'
        }
    
    return {
        'is_valid': True,
        'error_message': None,
        'error_type': None
    }


def validate_ticket_input(subject, description, ticket_id=None):
    """
    Validate complete ticket input (subject, description, and optionally ticket ID).
    
    Args:
        subject: Ticket subject
        description: Ticket description
        ticket_id: Optional ticket ID
        
    Returns:
        dict: {
            'is_valid': bool,
            'subject_error': dict or None,
            'description_error': dict or None,
            'ticket_id_error': dict or None,
            'all_errors': list of error messages
        }
    """
    subject_validation = validate_subject(subject)
    description_validation = validate_description(description)
    ticket_id_validation = None
    
    all_errors = []
    if not subject_validation['is_valid']:
        all_errors.append(subject_validation['error_message'])
    if not description_validation['is_valid']:
        all_errors.append(description_validation['error_message'])
    
    # Validate ticket ID if provided
    if ticket_id:
        ticket_id_validation = validate_ticket_id(ticket_id)
        if not ticket_id_validation['is_valid']:
            all_errors.append(ticket_id_validation['error_message'])
    
    return {
        'is_valid': subject_validation['is_valid'] and description_validation['is_valid'] and (ticket_id_validation['is_valid'] if ticket_id_validation else True),
        'subject_error': subject_validation if not subject_validation['is_valid'] else None,
        'description_error': description_validation if not description_validation['is_valid'] else None,
        'ticket_id_error': ticket_id_validation if ticket_id_validation and not ticket_id_validation['is_valid'] else None,
        'all_errors': all_errors
    }
