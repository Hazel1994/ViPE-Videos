import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv

def setup_gemini():
    """Initialize Gemini with API key from environment"""
    print("Initialize Gemini")
    load_dotenv()  # Load .env file
    api_key = os.getenv('GOOGLE_AI_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_AI_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

def call_gemini(model, prompt):
    """Basic Gemini API call"""
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        return response_text
    except Exception as e:
        print(f"Gemini API error: {e}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

def load_prompt_template(prompt_file_path=None):
    """
    Load the character generation prompt template from file
    """
    if prompt_file_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_file_path = os.path.join(os.path.dirname(current_dir), "prompts", "single_character_generation_prompt.txt")
    
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # Verify the template has the required placeholder
            if '{json_content}' not in content:
                print(f"Error: Prompt template missing {{json_content}} placeholder")
                return None
                
            return content
    except FileNotFoundError:
        print(f"Error: Prompt file not found: {prompt_file_path}")
        return None
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return None

def create_revision_prompt(original_response, additional_info):
    """
    Create a revision prompt with additional information - only revise prompts, keep characters unchanged
    """
    revision_prompt = f"""Based on the following additional information, please revise and improve ONLY the prompt interpretations. DO NOT change the characters in any way - keep them exactly as they are.

Only make changes to the prompts where the additional information indicates problems or improvements are needed. Fix misinterpretations or ambiguity in the prompts only. Don't explain why. Only use the information I provide.

Additional Information:
{additional_info}

Previous Response:
{original_response}

Please provide the revised version in the same format as before (two JSON blocks in markdown code blocks - first the updated prompts with your revisions, then the EXACT SAME characters without any changes)."""
    
    return revision_prompt

def get_user_approval_and_feedback(updated_prompts_path, characters_path):
    """
    Show user the generated content and get approval or additional feedback
    
    Returns:
        tuple: (approved: bool, additional_info: str or None)
    """
    print("\n" + "="*60)
    print("GEMINI INTERPRETATION REVIEW")
    print("="*60)
    
    # Show characters
    if characters_path and os.path.exists(characters_path):
        with open(characters_path, 'r') as f:
            characters = json.load(f)
        
        print(f"\nGenerated {len(characters)} characters:")
        for i, char in enumerate(characters, 1):
            print(f"  {i}. {char.get('name', 'Unknown')}")
            print(f"     Description: {char.get('description', 'No description')}")
            print()
    
    # Show sample of updated prompts
    if updated_prompts_path and os.path.exists(updated_prompts_path):
        with open(updated_prompts_path, 'r') as f:
            updated_prompts = json.load(f)
        
        print(f"Sample of updated prompts (showing first 3 of {len(updated_prompts)}):")
        for i, entry in enumerate(updated_prompts[:3]):
            start_time = entry.get('start', 'Unknown')
            prompt = entry.get('prompt', 'No prompt')
            print(f"  {i+1}. Time {start_time}s: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print()
    
    print("Do you approve this interpretation of your lyrics/story?")
    print("The characters and prompts will be used to generate your video.")
    
    while True:
        user_input = input("\nApprove interpretation? (y/n/help): ").lower().strip()
        
        if user_input in ['y', 'yes']:
            return True, None
        elif user_input in ['n', 'no']:
            print("\nPlease provide additional information to improve the interpretation:")
            print("Be specific about what should be changed, added, or corrected.")
            print("Press Enter twice to finish.")
            
            additional_info_lines = []
            while True:
                line = input()
                if line == "" and additional_info_lines and additional_info_lines[-1] == "":
                    break
                additional_info_lines.append(line)
            
            # Remove the last empty line
            if additional_info_lines and additional_info_lines[-1] == "":
                additional_info_lines.pop()
                
            additional_info = "\n".join(additional_info_lines).strip()
            
            if not additional_info:
                print("No additional information provided. Please try again.")
                continue
                
            return False, additional_info
        elif user_input == 'help':
            print("\nHelp:")
            print("y/yes - Approve the current interpretation and continue with video generation")
            print("n/no  - Provide additional information to improve the interpretation")
            print("help  - Show this help message")
        else:
            print("Please enter 'y', 'n', or 'help'")

def generate_characters(model, json_file_path, output_dir="./", prompt_file_path=None, logger=None):
    """
    Generate characters from lyrics interpretation file using Gemini with user approval loop
    
    Args:
        model: Gemini model instance
        json_file_path: Path to lyrics interpretation JSON (lyric2prompt)
        output_dir: Directory to save output files
        prompt_file_path: Path to custom prompt file (optional)
        logger: VideoGenerationLogger instance (optional)
        
    Returns:
        tuple: (success: bool, updated_prompts_path: str, characters_path: str)
    """
    
    try:
        # Read the JSON file
        with open(json_file_path, 'r') as f:
            lyrics_data = json.load(f)
        
        # Convert to string for the prompt
        json_content = json.dumps(lyrics_data, indent=2)
        
        # Load prompt template from file
        prompt_template = load_prompt_template(prompt_file_path)

        if prompt_template is None:
            print("Error: Failed to load prompt template")
            return False, None, None
        
        # Format the prompt with the JSON content
        initial_prompt = prompt_template.format(json_content=json_content)
        current_prompt = initial_prompt
        
        iteration = 0
        max_iterations = 3  # Prevent infinite loops
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\nCalling Gemini (attempt {iteration})...")
            
            # Call Gemini
            gemini_response = call_gemini(model, current_prompt)
            
            # Log the interaction
            if logger:
                logger.log_gemini_prompt_and_response(
                    current_prompt, 
                    gemini_response, 
                    success=(gemini_response is not None),
                    iteration=iteration
                )
            
            if gemini_response is None:
                print("Error: Failed to get response from Gemini")
                return False, None, None
            
            # Validate and save the response
            success, updated_prompts_path, characters_path = validate_and_save_gemini_response(
                gemini_response, json_file_path, output_dir
            )
            
            if not success:
                print("Error: Failed to validate Gemini response")
                if iteration < max_iterations:
                    print("Retrying with original prompt...")
                    current_prompt = initial_prompt
                    continue
                else:
                    return False, None, None
            
            # Get user approval
            approved, additional_info = get_user_approval_and_feedback(updated_prompts_path, characters_path)
            
            if approved:
                print("Interpretation approved by user!")
                return True, updated_prompts_path, characters_path
            
            if additional_info and iteration < max_iterations:
                print(f"User provided feedback, creating revision prompt...")
                current_prompt = create_revision_prompt(gemini_response, additional_info)
                
                # Log the additional information
                if logger:
                    logger.log_user_feedback(additional_info, iteration)
                    
                continue
            else:
                if iteration >= max_iterations:
                    print(f"Maximum iterations ({max_iterations}) reached.")
                    print("Using the last generated version.")
                    return True, updated_prompts_path, characters_path
                else:
                    print("No additional information provided, using current version.")
                    return True, updated_prompts_path, characters_path
        
        return False, None, None
        
    except Exception as e:
        if logger:
            logger.log_gemini_prompt_and_response(
                current_prompt if 'current_prompt' in locals() else "Failed to generate prompt", 
                None, 
                success=False, 
                error=e
            )
        print(f"ERROR in generate_characters: {e}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False, None, None

def validate_and_save_gemini_response(gemini_response, original_json_path, output_dir="./"):
    """
    Validate Gemini response and split into two JSON files
    
    Args:
        gemini_response: Raw response from Gemini
        original_json_path: Path to original JSON file (for fallback)
        output_dir: Directory to save output files
        
    Returns:
        tuple: (success: bool, updated_prompts_path: str, characters_path: str)
    """
    try:
        # First, try to extract JSON from markdown code blocks
        pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(pattern, gemini_response, re.DOTALL)
        
        updated_prompts = None
        characters = None
        
        if len(matches) == 2:
            # New format: Two separate JSON blocks in markdown
            print("Processing response with two JSON blocks in markdown")
            try:
                updated_prompts = json.loads(matches[0].strip())
                characters = json.loads(matches[1].strip())
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON in two-block markdown format: {e}")
                return False, None, None
                
        elif len(matches) == 1:
            # Old format: Single JSON block with characters at the end
            print("Processing response with single JSON block in markdown (legacy format)")
            try:
                cleaned_response = matches[0].strip()
                
                # Try to find the character list at the end
                character_pattern = r'\[\s*\{[^}]*"name"[^}]*"description"[^}]*\}[^\]]*\]'
                character_matches = list(re.finditer(character_pattern, cleaned_response, re.DOTALL))
                
                if not character_matches:
                    print("Error: No character list found in single block response")
                    return False, None, None
                
                # Get the last character list match
                last_character_match = character_matches[-1]
                character_json_str = last_character_match.group()
                
                # Remove character list from response to get updated prompts
                updated_prompts_str = cleaned_response[:last_character_match.start()].strip()
                
                # Parse both parts
                characters = json.loads(character_json_str)
                updated_prompts = json.loads(updated_prompts_str)
                
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON in single-block markdown format: {e}")
                return False, None, None
        else:
            # No markdown code blocks found - try to parse as raw JSON
            print("No markdown code blocks found, attempting to parse as raw JSON")
            
            # Look for two separate JSON arrays in the raw response
            json_array_pattern = r'\[\s*\{.*?\}\s*\]'
            json_arrays = re.findall(json_array_pattern, gemini_response, re.DOTALL)
            
            if len(json_arrays) == 2:
                print("Found two JSON arrays in raw response")
                try:
                    updated_prompts = json.loads(json_arrays[0])
                    characters = json.loads(json_arrays[1])
                except json.JSONDecodeError as e:
                    print(f"Error: Invalid JSON in raw format: {e}")
                    return False, None, None
            else:
                print(f"Error: Expected 2 JSON arrays in raw format, found {len(json_arrays)}")
                print("Available arrays:", [arr[:100] + "..." if len(arr) > 100 else arr for arr in json_arrays])
                return False, None, None
        
        # Validate updated prompts
        if not isinstance(updated_prompts, list):
            print("Error: Updated prompts must be a list")
            return False, None, None
            
        # Validate characters
        if not isinstance(characters, list):
            print("Error: Characters must be a list")
            return False, None, None
        
        if len(characters) > 5:
            print(f"Warning: Found {len(characters)} characters, expected max 5")
        
        # Validate character structure
        for char in characters:
            if not isinstance(char, dict) or "name" not in char or "description" not in char:
                print(f"Error: Invalid character structure: {char}")
                return False, None, None
                
        # Save files
        base_name = os.path.splitext(os.path.basename(original_json_path))[0]
        
        updated_prompts_path = os.path.join(output_dir, f"{base_name}_with_characters.json")
        characters_path = os.path.join(output_dir, f"{base_name}_characters.json")
        
        # Save updated prompts
        with open(updated_prompts_path, 'w') as f:
            json.dump(updated_prompts, f, indent=2)
        
        # Save characters
        with open(characters_path, 'w') as f:
            json.dump(characters, f, indent=2)
        
        print(f"Successfully saved:")
        print(f"  Updated prompts: {updated_prompts_path}")
        print(f"  Characters: {characters_path}")
        print(f"  Found {len(characters)} characters")
        
        return True, updated_prompts_path, characters_path
        
    except Exception as e:
        print(f"Error processing Gemini response: {e}")
        return False, None, None

def process_lyrics_with_characters(model, json_file_path, output_dir="./"):
    """
    Complete workflow: generate characters and save results
    
    Args:
        model: Gemini model instance
        json_file_path: Path to lyrics interpretation JSON
        output_dir: Directory to save output files
        
    Returns:
        tuple: (success: bool, updated_prompts_path: str, characters_path: str)
    """
    # Generate characters
    gemini_response = generate_characters_from_lyrics(model, json_file_path)
    
    if gemini_response is None:
        print("Error: Failed to get response from Gemini")
        return False, None, None
    
    # Validate and save
    return validate_and_save_gemini_response(gemini_response, json_file_path, output_dir)

